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
        style_network = dict(
            dim = 4,
            depth = 2
        ),
        image_size = 32,
        text_encoder = encoder
    ),
    discriminator = dict(
        image_size = 32,
        text_encoder = encoder
    ),
    amp = True
)

from src.models.gigagan.data import TextImageDataset

dataset = TextImageDataset(
    folder = '../datasets/laion2B-en-aesthetic',
    image_size = 32
)
dataloader = dataset.get_dataloader(batch_size = 1)
gan.set_dataloader(dataloader)

gan(
    steps = 10,
    grad_accum_every = 2
)