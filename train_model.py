import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('dataset/labels.xlsx')
print(f"Загружено {len(df)} документов")

class_mapping = {'Красный': 0, 'Желтый': 1, 'Зеленый': 2}
df['label'] = df['risk'].map(class_mapping)

#строки с пустым текстом
df = df.dropna(subset=['text'])
print(f"После очистки: {len(df)} документов")
print(f"Распределение: {df['risk'].value_counts().to_dict()}")

#ОБУЧАЮЩАЯ И ТЕСТОВАЯ ВЫБОРКИ
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)
print(f"Обучение: {len(X_train)} документов, Тест: {len(X_test)}")

#ТОКЕНИЗАТОР
MODEL_NAME = 'DeepPavlov/rubert-base-cased'
print(f"Загрузка модели {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

#МОДЕЛЬ
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

#ОБУЧЕНИЕ
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    report_to='none',
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()}
)

trainer.train()

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Модель сохранена в папку '{model_path}'")

results = trainer.evaluate()
print(f"Точность на тесте: {results['eval_accuracy']:.2%}")
