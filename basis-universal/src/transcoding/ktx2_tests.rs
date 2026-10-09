use crate::transcoding::{
    DecodeFlags, Ktx2Transcoder, TranscodeError, TranscoderTextureFormat,
};

/// ETC1S+BasisLZ KTX2 (what gltfpack -tc emits) decodes through the
/// ktx2_transcoder and transcodes into GPU block formats.
#[test]
fn ktx2_etc1s_basis_lz_transcodes() {
    let ktx2_file = include_bytes!("../../test_assets/rust-logo-etc1s-basis-lz.ktx2");

    let mut transcoder = Ktx2Transcoder::new();
    transcoder.init(ktx2_file).unwrap();

    assert!(transcoder.is_etc1s());
    assert_eq!(transcoder.width(), 64);
    assert_eq!(transcoder.height(), 64);
    assert_eq!(transcoder.dfd_transfer_func(), 2); // sRGB

    transcoder.start_transcoding().unwrap();

    // Raw RGBA32: 64 * 64 * 4 bytes, with actual image content (the rust
    // logo is transparent at the corners and red elsewhere, so not all
    // pixels are equal).
    let rgba = transcoder
        .transcode_image_level(0, TranscoderTextureFormat::RGBA32, DecodeFlags::empty())
        .unwrap();
    assert_eq!(rgba.len(), 64 * 64 * 4);
    assert!(!rgba.iter().all(|&b| b == 0));
    assert!(rgba.chunks_exact(4).any(|px| px[3] == 0));
    assert!(rgba.chunks_exact(4).any(|px| px[3] == 255));

    // BC7: (64/4) * (64/4) blocks * 16 bytes.
    let bc7 = transcoder
        .transcode_image_level(0, TranscoderTextureFormat::BC7_RGBA, DecodeFlags::empty())
        .unwrap();
    assert_eq!(bc7.len(), 16 * 16 * 16);
    assert!(!bc7.iter().all(|&b| b == 0));

    // All mip levels must transcode.
    for level_index in 0..transcoder.level_count() {
        transcoder
            .transcode_image_level(
                level_index,
                TranscoderTextureFormat::BC7_RGBA,
                DecodeFlags::empty(),
            )
            .unwrap();
    }
}

/// Raw (non-supercompressed) UASTC KTX2 decodes through the same wrapper.
#[test]
fn ktx2_uastc_transcodes() {
    let ktx2_file = include_bytes!("../../test_assets/rust-logo-uastc-zstd.ktx2");

    let mut transcoder = Ktx2Transcoder::new();
    transcoder.init(ktx2_file).unwrap();

    assert!(!transcoder.is_etc1s());
    assert_eq!(transcoder.width(), 64);
    transcoder.start_transcoding().unwrap();

    let bc7 = transcoder
        .transcode_image_level(0, TranscoderTextureFormat::BC7_RGBA, DecodeFlags::empty())
        .unwrap();
    assert_eq!(bc7.len(), 16 * 16 * 16);
}

/// Failure paths: garbage input must not panic.
#[test]
fn ktx2_rejects_garbage() {
    let mut transcoder = Ktx2Transcoder::new();
    assert!(matches!(
        transcoder.init(b"not a ktx2 file"),
        Err(TranscodeError::TranscodeFailed)
    ));
}
