//! Slide-ID to storage-address resolution.
//!
//! Ported from `SlideManager._get_slide_storage_addresses` (`slide_manager.py:94`) and the
//! local-mapper interface in `wsi_service/base_mapper.py`.

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::{ApiError, Result};
use crate::models::{CaseLocalMapper, SlideLocalMapper, StorageAddress};

/// `BaseMapper` — `wsi_service/base_mapper.py:7`.
///
/// Mappers are synchronous in Python and do blocking filesystem work; keep them
/// synchronous here too and let callers wrap them in `spawn_blocking` where it matters.
pub trait LocalMapper: Send + Sync {
    /// Only `PathsMapper` sets this. Context-dependent mappers require `?context=` on
    /// `/cases/` and cannot enumerate without it.
    fn is_context_dependent(&self) -> bool {
        false
    }

    fn get_cases(&self, context: Option<&str>) -> Result<Vec<CaseLocalMapper>>;
    fn get_slides(&self, case_id: &str) -> Result<Vec<SlideLocalMapper>>;
    fn get_slide(&self, slide_id: &str) -> Result<SlideLocalMapper>;

    /// Context-independent mappers rebuild their index here; the others no-op.
    fn refresh(&self, _force: bool) -> Result<()> {
        Ok(())
    }

    fn load(&self) -> Result<()> {
        Ok(())
    }
}

/// The JSON the remote storage-mapper service returns.
#[derive(Debug, Deserialize)]
struct RemoteSlideStorage {
    storage_addresses: Vec<StorageAddress>,
}

/// Resolves slide IDs to storage addresses, either from a local mapper or over HTTP.
pub enum StorageResolver {
    Local(std::sync::Arc<dyn LocalMapper>),
    Remote {
        client: reqwest::Client,
        /// Template containing `{slide_id}`.
        mapper_address: String,
    },
}

impl StorageResolver {
    pub fn remote(client: reqwest::Client, mapper_address: impl Into<String>) -> Self {
        Self::Remote {
            client,
            mapper_address: mapper_address.into(),
        }
    }

    pub fn local(mapper: std::sync::Arc<dyn LocalMapper>) -> Self {
        Self::Local(mapper)
    }

    /// `_get_slide_storage_addresses` — `slide_manager.py:94`.
    pub async fn storage_addresses(&self, slide_id: &str) -> Result<Vec<StorageAddress>> {
        match self {
            StorageResolver::Local(mapper) => {
                // The mapper raises its own 404 for an unknown slide; Python's extra
                // `if not slide` guard after that call is unreachable.
                let slide = mapper.get_slide(slide_id)?;
                Ok(slide.slide_storage.storage_addresses)
            }
            StorageResolver::Remote {
                client,
                mapper_address,
            } => {
                let url = mapper_address.replace("{slide_id}", slide_id);
                // Python catches only `aiohttp.ClientConnectorError` and turns it into a
                // 503. Any transport failure lands here, which is the same outcome for
                // every case a client can distinguish.
                let response = client.get(&url).send().await.map_err(|_| {
                    ApiError::service_unavailable(
                        "WSI Service is unable to connect to the Storage Mapper Service.",
                    )
                })?;

                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    return Err(ApiError::not_found(format!(
                        "Could not find a storage address for slide id {slide_id}."
                    )));
                }

                // Python calls `r.json()` unconditionally for any non-404 status, so a 5xx
                // body that is not JSON surfaces as a decode failure, not as the upstream
                // status. Reproduced.
                let parsed: RemoteSlideStorage = response.json().await.map_err(|err| {
                    ApiError::internal(format!("Storage mapper returned an unreadable body: {err}"))
                })?;
                Ok(parsed.storage_addresses)
            }
        }
    }

    /// `_get_slide_main_storage_address` — `slide_manager.py:118`.
    /// First address flagged as main, else the first address.
    pub async fn main_storage_address(&self, slide_id: &str) -> Result<StorageAddress> {
        let addresses = self.storage_addresses(slide_id).await?;
        if let Some(main) = addresses.iter().find(|a| a.main_address) {
            return Ok(main.clone());
        }
        addresses.into_iter().next().ok_or_else(|| {
            // `storage_addresses[0]` on an empty list raises IndexError, which FastAPI
            // does not handle.
            ApiError::unhandled("IndexError('list index out of range')")
        })
    }
}

#[async_trait]
pub trait StorageResolverExt {
    async fn file_paths(&self, data_dir: &str, slide_id: &str) -> Result<Vec<String>>;
}

#[async_trait]
impl StorageResolverExt for StorageResolver {
    /// `get_slide_file_paths` — `slide_manager.py:74`.
    async fn file_paths(&self, data_dir: &str, slide_id: &str) -> Result<Vec<String>> {
        let addresses = self.storage_addresses(slide_id).await?;
        Ok(addresses
            .into_iter()
            .map(|a| join_data_dir(data_dir, &a.address))
            .collect())
    }
}

/// `os.path.join(self.data_dir, address)`.
///
/// `os.path.join` returns the second argument unchanged when it is absolute, and the
/// mappers do emit absolute addresses in some configurations, so that behaviour is kept.
pub fn join_data_dir(data_dir: &str, address: &str) -> String {
    let path = std::path::Path::new(address);
    if path.is_absolute() {
        return address.to_owned();
    }
    let mut base = data_dir.trim_end_matches('/').to_owned();
    base.push('/');
    base.push_str(address.trim_start_matches('/'));
    base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_addresses_are_joined_under_the_data_dir() {
        assert_eq!(
            join_data_dir("/data", "case/slide.svs"),
            "/data/case/slide.svs"
        );
        assert_eq!(
            join_data_dir("/data/", "case/slide.svs"),
            "/data/case/slide.svs"
        );
        assert_eq!(join_data_dir("/data", "/case/slide.svs"), "/case/slide.svs");
    }

    #[test]
    fn absolute_addresses_win_like_os_path_join() {
        // Only meaningful on platforms where the address parses as absolute.
        let absolute = if cfg!(windows) {
            "C:/x/y.svs"
        } else {
            "/x/y.svs"
        };
        assert_eq!(join_data_dir("/data", absolute), absolute);
    }
}
