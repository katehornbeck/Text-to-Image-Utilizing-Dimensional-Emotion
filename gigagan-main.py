from src.models.gigagan import GigaGAN
from src.models.gigagan.text_encoder import TextEncoder

text_encoder = dict(
    dim = 64,
    depth = 120,
    vad_model_path = "/home/kate/grad_school/GenAI/final/data/private/try-2/ckpt/trained/emobank-vad-regression-7560-30.ckpt",
    vad_config_path = "/home/kate/grad_school/GenAI/final/config-inference.txt"
)

encoder = TextEncoder(**text_encoder)

gan = GigaGAN(
    generator = dict(
        dim_capacity = 8,
        style_network = dict(
            dim = 64,
            depth = 4
        ),
        image_size = 256,
        dim_max = 512,
        num_skip_layers_excite = 4,
        text_encoder = encoder
    ),
    discriminator = dict(
        dim_capacity = 16,
        dim_max = 512,
        image_size = 256,
        num_skip_layers_excite = 4,
        text_encoder = encoder
    ),
    amp = True
)

from src.models.gigagan.data import TextImageDataset

dataset = TextImageDataset(
    folder = '../datasets/laion2B-en-aesthetic',
    image_size = 256
)
dataloader = dataset.get_dataloader(batch_size = 1)
gan.set_dataloader(dataloader)

gan(
    steps = 100,
    grad_accum_every = 8
)