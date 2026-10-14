//! A forward-only ZIP writer.
//!
//! `/slides/download` streams the archive: a whole-slide image can be tens of gigabytes,
//! so neither buffering it in memory nor staging it on disk is acceptable. The `zip`
//! crate's writer needs `Seek` because it patches sizes back into the local header after
//! writing each entry, which a stream cannot do.
//!
//! The format solves this with **data descriptors** — general-purpose flag bit 3 — which
//! move the CRC and sizes to *after* the entry data. That is the same mechanism `zipfly`
//! uses on the Python side. Entries are stored uncompressed, matching `ZIP_STORED`:
//! slide data is already compressed, so deflating it would only burn CPU.
//!
//! ZIP64 is used unconditionally, because a single `.svs` routinely exceeds the 4 GiB
//! limit of the classic format.

use std::io::{Read, Write};

const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const DATA_DESCRIPTOR_SIGNATURE: u32 = 0x0807_4b50;
const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const ZIP64_EOCD_SIGNATURE: u32 = 0x0606_4b50;
const ZIP64_EOCD_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const EOCD_SIGNATURE: u32 = 0x0605_4b50;

const ZIP64_EXTRA_FIELD_ID: u16 = 0x0001;

/// 4.5 — the minimum version that understands ZIP64.
const VERSION_NEEDED: u16 = 45;
/// Bit 3: sizes and CRC follow the data. Bit 11: the name is UTF-8.
const FLAG_DATA_DESCRIPTOR: u16 = 0x0008;
const FLAG_UTF8_NAME: u16 = 0x0800;
const METHOD_STORED: u16 = 0;

/// Placeholders that tell a reader to look in the ZIP64 extra field instead.
const U32_SENTINEL: u32 = 0xFFFF_FFFF;
const U16_SENTINEL: u16 = 0xFFFF;

struct CentralEntry {
    name: String,
    crc32: u32,
    size: u64,
    local_header_offset: u64,
}

pub struct StreamingZip<W: Write> {
    out: W,
    offset: u64,
    entries: Vec<CentralEntry>,
}

impl<W: Write> StreamingZip<W> {
    pub fn new(out: W) -> Self {
        Self {
            out,
            offset: 0,
            entries: Vec::new(),
        }
    }

    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.out.write_all(bytes)?;
        self.offset += bytes.len() as u64;
        Ok(())
    }

    /// Appends one stored entry, streaming `source` straight through.
    pub fn add_entry<R: Read>(&mut self, name: &str, source: &mut R) -> std::io::Result<()> {
        let local_header_offset = self.offset;
        let name_bytes = name.as_bytes();

        // --- local file header ---------------------------------------------------------
        let mut header = Vec::with_capacity(30 + name_bytes.len() + 20);
        header.extend_from_slice(&LOCAL_HEADER_SIGNATURE.to_le_bytes());
        header.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
        header.extend_from_slice(&(FLAG_DATA_DESCRIPTOR | FLAG_UTF8_NAME).to_le_bytes());
        header.extend_from_slice(&METHOD_STORED.to_le_bytes());
        // Modification time and date are zeroed. Python writes the file's real mtime, so
        // archives are not byte-identical; the entry contents are.
        header.extend_from_slice(&0u16.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        // CRC and sizes are unknown until the data has been written; the data descriptor
        // carries the real values.
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&U32_SENTINEL.to_le_bytes());
        header.extend_from_slice(&U32_SENTINEL.to_le_bytes());
        header.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        // ZIP64 extra field: id, payload length, then the two 64-bit sizes.
        header.extend_from_slice(&20u16.to_le_bytes());
        header.extend_from_slice(name_bytes);
        header.extend_from_slice(&ZIP64_EXTRA_FIELD_ID.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        self.write_all(&header)?;

        // --- file data -----------------------------------------------------------------
        let mut hasher = crc32fast::Hasher::new();
        let mut size = 0u64;
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            size += read as u64;
            self.write_all(&buffer[..read])?;
        }
        let crc32 = hasher.finalize();

        // --- data descriptor -----------------------------------------------------------
        let mut descriptor = Vec::with_capacity(24);
        descriptor.extend_from_slice(&DATA_DESCRIPTOR_SIGNATURE.to_le_bytes());
        descriptor.extend_from_slice(&crc32.to_le_bytes());
        // ZIP64: both sizes are 8 bytes here. Stored, so they are equal.
        descriptor.extend_from_slice(&size.to_le_bytes());
        descriptor.extend_from_slice(&size.to_le_bytes());
        self.write_all(&descriptor)?;

        self.entries.push(CentralEntry {
            name: name.to_owned(),
            crc32,
            size,
            local_header_offset,
        });
        Ok(())
    }

    /// Writes the central directory and both end-of-archive records.
    pub fn finish(mut self) -> std::io::Result<W> {
        let central_directory_offset = self.offset;

        for index in 0..self.entries.len() {
            let (name, crc32, size, local_header_offset) = {
                let entry = &self.entries[index];
                (
                    entry.name.clone(),
                    entry.crc32,
                    entry.size,
                    entry.local_header_offset,
                )
            };
            let name_bytes = name.as_bytes();

            let mut record = Vec::with_capacity(46 + name_bytes.len() + 28);
            record.extend_from_slice(&CENTRAL_HEADER_SIGNATURE.to_le_bytes());
            // Version made by: 4.5, MS-DOS host (0).
            record.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
            record.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
            record.extend_from_slice(&(FLAG_DATA_DESCRIPTOR | FLAG_UTF8_NAME).to_le_bytes());
            record.extend_from_slice(&METHOD_STORED.to_le_bytes());
            record.extend_from_slice(&0u16.to_le_bytes());
            record.extend_from_slice(&0u16.to_le_bytes());
            record.extend_from_slice(&crc32.to_le_bytes());
            record.extend_from_slice(&U32_SENTINEL.to_le_bytes());
            record.extend_from_slice(&U32_SENTINEL.to_le_bytes());
            record.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            // ZIP64 extra: uncompressed size, compressed size, local header offset.
            record.extend_from_slice(&28u16.to_le_bytes());
            record.extend_from_slice(&0u16.to_le_bytes()); // comment length
            record.extend_from_slice(&0u16.to_le_bytes()); // disk number start
            record.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
            record.extend_from_slice(&0u32.to_le_bytes()); // external attributes
            record.extend_from_slice(&U32_SENTINEL.to_le_bytes()); // local header offset
            record.extend_from_slice(name_bytes);
            record.extend_from_slice(&ZIP64_EXTRA_FIELD_ID.to_le_bytes());
            record.extend_from_slice(&24u16.to_le_bytes());
            record.extend_from_slice(&size.to_le_bytes());
            record.extend_from_slice(&size.to_le_bytes());
            record.extend_from_slice(&local_header_offset.to_le_bytes());
            self.write_all(&record)?;
        }

        let central_directory_size = self.offset - central_directory_offset;
        let entry_count = self.entries.len() as u64;
        let zip64_eocd_offset = self.offset;

        // --- ZIP64 end of central directory ------------------------------------------
        let mut zip64_eocd = Vec::with_capacity(56);
        zip64_eocd.extend_from_slice(&ZIP64_EOCD_SIGNATURE.to_le_bytes());
        // Size of this record, excluding the signature and this field: 56 - 12.
        zip64_eocd.extend_from_slice(&44u64.to_le_bytes());
        zip64_eocd.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
        zip64_eocd.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
        zip64_eocd.extend_from_slice(&0u32.to_le_bytes()); // this disk
        zip64_eocd.extend_from_slice(&0u32.to_le_bytes()); // disk with the directory
        zip64_eocd.extend_from_slice(&entry_count.to_le_bytes());
        zip64_eocd.extend_from_slice(&entry_count.to_le_bytes());
        zip64_eocd.extend_from_slice(&central_directory_size.to_le_bytes());
        zip64_eocd.extend_from_slice(&central_directory_offset.to_le_bytes());
        self.write_all(&zip64_eocd)?;

        // --- ZIP64 locator -------------------------------------------------------------
        let mut locator = Vec::with_capacity(20);
        locator.extend_from_slice(&ZIP64_EOCD_LOCATOR_SIGNATURE.to_le_bytes());
        locator.extend_from_slice(&0u32.to_le_bytes());
        locator.extend_from_slice(&zip64_eocd_offset.to_le_bytes());
        locator.extend_from_slice(&1u32.to_le_bytes());
        self.write_all(&locator)?;

        // --- end of central directory --------------------------------------------------
        let mut eocd = Vec::with_capacity(22);
        eocd.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes());
        eocd.extend_from_slice(&U16_SENTINEL.to_le_bytes());
        eocd.extend_from_slice(&U16_SENTINEL.to_le_bytes());
        eocd.extend_from_slice(&U32_SENTINEL.to_le_bytes());
        eocd.extend_from_slice(&U32_SENTINEL.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes());
        self.write_all(&eocd)?;

        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut zip = StreamingZip::new(Vec::new());
        for (name, data) in entries {
            zip.add_entry(name, &mut Cursor::new(data.to_vec()))
                .unwrap();
        }
        zip.finish().unwrap()
    }

    /// Reads the archive back with an independent implementation, so the test checks the
    /// bytes rather than the writer's own assumptions.
    fn read_back(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes.to_vec())).unwrap();
        let mut out = Vec::new();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let name = file.name().to_owned();
            let mut data = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut data).unwrap();
            out.push((name, data));
        }
        out
    }

    #[test]
    fn a_single_entry_round_trips() {
        let bytes = build(&[("a.txt", b"hello")]);
        let entries = read_back(&bytes);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "a.txt");
        assert_eq!(entries[0].1, b"hello");
    }

    #[test]
    fn multiple_entries_keep_their_order_and_contents() {
        let bytes = build(&[
            ("slide/a.dcm", b"first"),
            ("slide/b.dcm", b"second"),
            ("slide.mrxs", b"third"),
        ]);
        let entries = read_back(&bytes);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, "slide/a.dcm");
        assert_eq!(entries[0].1, b"first");
        assert_eq!(entries[2].0, "slide.mrxs");
        assert_eq!(entries[2].1, b"third");
    }

    #[test]
    fn an_empty_entry_is_valid() {
        let bytes = build(&[("empty.bin", b"")]);
        let entries = read_back(&bytes);
        assert_eq!(entries[0].1.len(), 0);
    }

    #[test]
    fn an_empty_archive_is_valid() {
        let bytes = build(&[]);
        let archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn entries_are_stored_not_deflated() {
        let payload = vec![b'a'; 4096];
        let bytes = build(&[("big.txt", &payload)]);
        // Stored data appears verbatim in the stream.
        assert!(bytes
            .windows(payload.len())
            .any(|window| window == payload.as_slice()));
    }

    #[test]
    fn utf8_names_survive() {
        let bytes = build(&[("případ/řez.svs", b"x")]);
        let entries = read_back(&bytes);
        assert_eq!(entries[0].0, "případ/řez.svs");
    }

    #[test]
    fn crc_is_computed_over_the_entry_data() {
        // A wrong CRC makes the reader fail while reading, so a successful round trip of a
        // multi-chunk payload is the check that matters.
        let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let bytes = build(&[("payload.bin", &payload)]);
        let entries = read_back(&bytes);
        assert_eq!(entries[0].1, payload);
    }
}
