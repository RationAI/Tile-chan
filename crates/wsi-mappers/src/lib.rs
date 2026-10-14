//! Local mappers: slide-ID to filesystem resolution without a storage-mapper service.
//!
//! Selected by `WS_LOCAL_MODE`, which carries a Python `module:Class` descriptor. The
//! existing descriptors are accepted verbatim so deployed `.env` files keep working; see
//! [`from_descriptor`].

use std::sync::Arc;

use wsi_core::error::{ApiError, Result};
use wsi_core::plugins::PluginRegistry;
use wsi_core::storage::LocalMapper;

pub mod csv;
pub mod iterator;
pub mod paths;
pub mod secondary;
pub mod simple;

pub use csv::{CsvMapper, CsvMapperSettings};
pub use iterator::{IteratorMapper, IteratorSettings};
pub use paths::PathsMapper;
pub use secondary::{abs_file_path_to_relative, collect_secondary_files};
pub use simple::{uuid5_url_hex, SimpleMapper};

/// Builds a mapper from a `WS_LOCAL_MODE` descriptor.
///
/// The Python values are `module:Class` paths; they are matched exactly so an unchanged
/// `.env` selects the same mapper. An unknown descriptor is a startup error rather than a
/// silent fallback — running with the wrong mapper would hand out wrong slide IDs.
pub fn from_descriptor(
    descriptor: &str,
    data_dir: &str,
    registry: Arc<PluginRegistry>,
) -> Result<Arc<dyn LocalMapper>> {
    match descriptor {
        "wsi_service.paths_mapper:PathsMapper" | "paths" => {
            Ok(Arc::new(PathsMapper::new(data_dir, registry)))
        }
        "wsi_service.simple_mapper:SimpleMapper" | "simple" => {
            Ok(Arc::new(SimpleMapper::new(data_dir, registry)?))
        }
        // The CSV and iterator mappers read their own `CSWS_`/`WSIT_` settings rather than
        // taking configuration through this call, matching their Python constructors.
        "wsi_service.csv_mapper:CSVMapper" | "csv" => Ok(csv::from_env(data_dir)?),
        "wsi_service.mapper_iterator.iterator:IteratorMapper" | "iterator" => {
            Ok(iterator::from_env(data_dir)?)
        }
        other => Err(ApiError::internal(format!(
            "Unknown WS_LOCAL_MODE descriptor: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Arc<PluginRegistry> {
        Arc::new(PluginRegistry::new(vec![]))
    }

    #[test]
    fn python_descriptors_resolve() {
        let dir = std::env::temp_dir().to_string_lossy().replace('\\', "/");
        assert!(from_descriptor("wsi_service.paths_mapper:PathsMapper", &dir, registry()).is_ok());
        assert!(from_descriptor("paths", &dir, registry()).is_ok());
    }

    #[test]
    fn unknown_descriptor_is_an_error() {
        let dir = std::env::temp_dir().to_string_lossy().replace('\\', "/");
        // `unwrap_err` needs `T: Debug`, which `Arc<dyn LocalMapper>` cannot provide.
        let err = match from_descriptor("some.module:Thing", &dir, registry()) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert!(err
            .detail_text()
            .contains("Unknown WS_LOCAL_MODE descriptor"));
    }
}
