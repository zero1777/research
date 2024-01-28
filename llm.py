import os
import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import pickle
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torch.nn import CrossEntropyLoss

#tensorboard --logdir logs/ --host localhost --port 8088


class QADatasetjson(Dataset):
    """
    A custom dataset class for question answering data.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["instruction"] + " " + item["input"], item["output"], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        start_positions = inputs["start_positions"].squeeze()
        end_positions = inputs["end_positions"].squeeze()
        return input_ids, attention_mask, start_positions, end_positions
    
    
class QADatasetold(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['instruction'],
            item['input'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        # Find the start and end character positions of the answer span
        start_pos = item['text'].find(item['output'])
        end_pos = start_pos + len(item['output'])
        
        start_positions = encoding.char_to_token(start_pos)
        end_positions = encoding.char_to_token(end_pos)
        
        # Handle cases where char_to_token returns None
        if start_positions is None:
            start_positions = 0
        if end_positions is None:
            end_positions = encoding['input_ids'].shape[1] - 1
        
        encoding.update({
            'start_positions': start_positions,
            'end_positions': end_positions
        })
        return encoding





class CustomDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}


class trainbase:
    def extract_text_from_pdfs(self, pdf_folder):
        text = ""
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                pdf_file = os.path.join(pdf_folder, filename)
                with open(pdf_file, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in range(len(pdf_reader.pages)):
                        page_obj = pdf_reader.pages[page]
                        text += page_obj.extract_text()
        return text

    def train_model(self, pdf_folder, model_name, batch_size, max_length):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        text = self.extract_text_from_pdfs(pdf_folder)
        with open("Output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)
        paragraphs = text.split('\n')
        with open('paragraphs.txt', 'w',encoding="utf-8") as f:
            for paragraph in paragraphs:
                f.write(f'{paragraph}\n')
        list_length = len(paragraphs)

        tokenized_paragraphs = [tokenizer.encode(paragraph) for paragraph in paragraphs]
        input_ids = []
        labels = []
        current_input_ids = []
        for tokenized_paragraph in tokenized_paragraphs:
            current_input_ids.extend(tokenized_paragraph)
            while len(current_input_ids) >= max_length:
                input_ids.append(current_input_ids[:max_length])
                labels.append(current_input_ids[:max_length])
                current_input_ids = current_input_ids[max_length:]  
        max_length = min(max_length, tokenizer.model_max_length)
        input_ids = [ids[:max_length] for ids in input_ids]
        labels = [lbls[:max_length] for lbls in labels]
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        print(input_ids.shape, input_ids.dtype)
        print(input_ids[:100])
        decoded_text = tokenizer.decode(input_ids[0])
        dataset = CustomDataset(input_ids, labels)
        eval_index = int(len(dataset) * 0.9)
        train_dataset = CustomDataset(input_ids[:eval_index], labels[:eval_index])
        eval_dataset = CustomDataset(input_ids[eval_index:], labels[eval_index:])
        
 
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            logging_dir="./logs",
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.train()
        trainer.save_model("./outputmodel/")
        tokenizer.save_pretrained("./outputmodel/")


class bart_text_Generation:
    def __init__(self, model_name):
        from transformers import BartTokenizer, BartForConditionalGeneration

        # Load the tokenizer and fine-tuned model
        tokenizer = BartTokenizer.from_pretrained('bart-base')
        model = BartForConditionalGeneration.from_pretrained('./results')

        # Generate text
        input_text = "Insert some input text here"
        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model.generate(inputs.input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(generated_text)



class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['instruction'] + ' ' + item['input'], return_tensors='pt', padding='max_length', truncation=True)
        start_positions = item['output'][0]
        end_positions = item['output'][1]
        return {
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'start_positions': start_positions,
            'end_positions': end_positions
        }

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        loss_fn = CrossEntropyLoss()
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"].view(-1)
        end_positions = inputs["end_positions"].view(-1)
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

class finetune():
    def fine_tune_model(self):
            
        # tokenizer = BartTokenizer.from_pretrained("./outputmodel/")
        # model = BartForConditionalGeneration.from_pretrained("./results/checkpoint-4500")

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  

        if os.path.exists('processed_data.pkl'):
            with open('processed_data.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            datasetload = load_dataset("vicgalle/alpaca-gpt4")
            dataset = datasetload['train']
            data = []
            for i in tqdm(range(1000), desc="Converting dataset"):
                start_pos = dataset['text'][i].find(dataset['output'][i])
                end_pos = start_pos + len(dataset['output'][i])
                item = {
                    'instruction': dataset['instruction'][i],
                    'input': dataset['input'][i],
                    'output': [start_pos, end_pos],
                    'text': dataset['text'][i]
                }
                data.append(item)
            with open('processed_data.pkl', 'wb') as f:
                pickle.dump(data, f)
        split_index = int(len(data) * 0.9)
        train_dataset = data[:split_index]
        test_dataset = data[split_index:]
        train_dataset = QADataset(train_dataset, tokenizer)
        test_dataset = QADataset(test_dataset, tokenizer)
        
        batch_size = 1

        training_args = TrainingArguments(
            output_dir="./resultsqa",
            evaluation_strategy="epoch",
            logging_dir="./logsqa",
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
        )
        
        
        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=self.custom_data_collator,
        )
        trainer.train()
        trainer.save_model("./outputmodelstage2/")
        tokenizer.save_pretrained("./outputmodelstage2/")

        
    def custom_data_collator(self, features):
        batch = default_data_collator(features)
        batch['start_positions'] = torch.tensor([f['start_positions'] for f in features])
        batch['end_positions'] = torch.tensor([f['end_positions'] for f in features])
        return batch

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        loss_fn = CrossEntropyLoss()
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"].view(-1)
        end_positions = inputs["end_positions"].view(-1)
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

class BERTQA:
    def __init__(self):
        # Load the tokenizer and fine-tuned model
        tokenizer = BartTokenizer.from_pretrained("./outputmodelstage2/")
        model = BartForConditionalGeneration.from_pretrained("./outputmodelstage2/")

        # Answer a question
        question = "What is the Design Criteria for Bridges and Other structures"
        context = ""
        inputs = tokenizer(question + ' ' + context, return_tensors='pt')
        outputs = model.generate(inputs.input_ids)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(answer)




def main():
    
    #In this example, we specify "gpt2" as the value of the model_name argument to load and train a GPT-2 model. You can replace "gpt2" with the name of any other causal language model available in the transformers library, such as "gpt3" or "xlnet".
    # trainer = trainbase()
    # trainer.train_model(pdf_folder='./inputs/', model_name='facebook/bart-base', batch_size=1, max_length=512)

    trainer = finetune()
    trainer.fine_tune_model()
    return
    
if __name__ == "__main__":
    main()