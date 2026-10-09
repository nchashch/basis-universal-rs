use crate::sys;
use crate::transcoding::transcoder_init;
use crate::transcoding::{DecodeFlags, TranscodeError, TranscoderTextureFormat};

/// Safe wrapper around the C++ `basisu::ktx2_transcoder`, which decodes KTX2
/// files containing ETC1S data — with or without BasisLZ supercompression
/// (the scheme `gltfpack -tc` emits, which the `ktx2` crate cannot decode) —
/// or UASTC data (raw or Zstd/Zlib supercompressed), transcoding into GPU
/// block formats.
pub struct Ktx2Transcoder {
    ptr: *mut sys::Ktx2Transcoder,

    /// `ktx2_transcoder::init()` keeps a pointer to the file data until the
    /// object is destroyed, so the wrapper owns a copy of it.
    data: Vec<u8>,
}

impl Ktx2Transcoder {
    /// Create a Ktx2Transcoder
    pub fn new() -> Ktx2Transcoder {
        transcoder_init();
        Self {
            ptr: unsafe { sys::ktx2_transcoder_new() },
            data: Vec::new(),
        }
    }

    /// Parse the KTX2 header, level index array, DFD and key values.
    pub fn init(&mut self, data: &[u8]) -> Result<(), TranscodeError> {
        if data.len() > u32::MAX as usize {
            return Err(TranscodeError::TranscodeFailed);
        }
        self.data = data.to_vec();
        if unsafe {
            sys::ktx2_transcoder_init(
                self.ptr,
                self.data.as_ptr() as *const _,
                self.data.len() as u32,
            )
        } {
            Ok(())
        } else {
            Err(TranscodeError::TranscodeFailed)
        }
    }

    /// Texture width in texels.
    pub fn width(&self) -> u32 {
        unsafe { sys::ktx2_transcoder_get_width(self.ptr) }
    }

    /// Texture height in texels.
    pub fn height(&self) -> u32 {
        unsafe { sys::ktx2_transcoder_get_height(self.ptr) }
    }

    /// Number of mipmap levels (always 1 or more).
    pub fn level_count(&self) -> u32 {
        unsafe { sys::ktx2_transcoder_get_levels(self.ptr) }
    }

    /// True if the file holds ETC1S data, false for UASTC.
    pub fn is_etc1s(&self) -> bool {
        unsafe { sys::ktx2_transcoder_is_etc1s(self.ptr) }
    }

    /// True if the ETC1S file has two planes (typically RGBA or RRRG).
    pub fn has_alpha(&self) -> bool {
        unsafe { sys::ktx2_transcoder_get_has_alpha(self.ptr) != 0 }
    }

    /// The DFD transfer function: `1` for KTX2_KHR_DF_TRANSFER_LINEAR, `2`
    /// for KTX2_KHR_DF_TRANSFER_SRGB.
    pub fn dfd_transfer_func(&self) -> u32 {
        unsafe { sys::ktx2_transcoder_get_dfd_transfer_func(self.ptr) }
    }

    /// Must be called before [`Self::transcode_image_level`]. For ETC1S files
    /// this decompresses the global endpoint/selector codebooks (the BasisLZ
    /// supercompression global data), which is expensive — call it once per
    /// file.
    pub fn start_transcoding(&mut self) -> Result<(), TranscodeError> {
        if unsafe { sys::ktx2_transcoder_start_transcoding(self.ptr) } {
            Ok(())
        } else {
            Err(TranscodeError::TranscodeFailed)
        }
    }

    /// Transcode a single mipmap level (layer 0, face 0) into `format`.
    pub fn transcode_image_level(
        &mut self,
        level_index: u32,
        format: TranscoderTextureFormat,
        decode_flags: DecodeFlags,
    ) -> Result<Vec<u8>, TranscodeError> {
        if level_index >= self.level_count() {
            return Err(TranscodeError::ImageLevelNotFound);
        }
        // Level dimensions shift down by the level index, minimum 1.
        let level_width = (self.width() >> level_index).max(1);
        let level_height = (self.height() >> level_index).max(1);
        let total_blocks = format.calculate_minimum_output_buffer_blocks_or_pixels(
            level_width,
            level_height,
            level_width.div_ceil(4) * level_height.div_ceil(4),
            None,
            None,
        );
        let required_buffer_bytes = format.calculate_minimum_output_buffer_bytes(
            level_width,
            level_height,
            total_blocks,
            None,
            None,
        );

        let mut output = vec![0_u8; required_buffer_bytes as usize];
        if unsafe {
            sys::ktx2_transcoder_transcode_image_level(
                self.ptr,
                level_index,
                output.as_mut_ptr() as *mut _,
                total_blocks,
                format.into(),
                decode_flags.bits(),
            )
        } {
            Ok(output)
        } else {
            Err(TranscodeError::TranscodeFailed)
        }
    }
}

impl Default for Ktx2Transcoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Ktx2Transcoder {
    fn drop(&mut self) {
        unsafe {
            sys::ktx2_transcoder_delete(self.ptr);
        }
    }
}
