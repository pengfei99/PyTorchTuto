from transformers import CamembertModel, CamembertTokenizer
from transformers import pipeline

# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

camembert_fill_mask = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
print(results)
