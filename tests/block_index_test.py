from dncformer.config import CFG
from dncformer.train.loop import load_base_model
from dncformer.model.head import DNCFormerHead


device = "cuda"

CFG.per_block_cfg = [
    {"N": 24, "W": 8,  "R": 1},
    {"N": 16, "W": 16, "R": 2},
]
tok, base = load_base_model(CFG.base_model_id)
head = DNCFormerHead(base, CFG).to(device)
assert head.blocks[0].dncblocks[0].N == 24 and head.blocks[0].dncblocks[0].W == 8
assert head.blocks[1].dncblocks[0].N == 16 and head.blocks[1].dncblocks[0].W == 16
print("per_block_cfg applied correctly.")