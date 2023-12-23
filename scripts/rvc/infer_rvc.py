from test_infer import voice_conversion


voice_conversion(
    f0up_key=0,
    input_path='audio.ogg',
    index_path='D:\\Dev\\RVC_BOT\\MODELS\\Стримеры и блогеры\\Каша\\kussia.index',
    f0method='rmvpe',
    opt_path='audio_out.mp3',
    model_path='D:\\Dev\\RVC_BOT\\MODELS\\Стримеры и блогеры\\Каша\\kussia.pth',
    index_rate=0.8,
    device='cuda:0',
    is_half=True,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1,
    protect=0.33,
    crepe_hop_length=128,
    f0_minimum=50,
    f0_maximum=1100,
    autotune_enable=False,
)