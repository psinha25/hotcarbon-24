import random
import os
import pathlib
from abc import ABC, abstractmethod
import torch
import torchaudio                       # type: ignore
from datasets import load_dataset       # type: ignore
from diffusers import DiffusionPipeline # type: ignore
from transformers import (              # type: ignore
    BertTokenizer,
    BertForMaskedLM,
    GPTJForCausalLM,
    AutoTokenizer,
    pipeline
)

class Inference(ABC):
    def __init__(self, model_name, device_id, batch_size):
        self._device = torch.device(f"cuda:{device_id}")
        self._model_name = model_name
        self._model = None
        self._batch_size = batch_size
    
    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def infer(self):
        pass

class StableDiffusion(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._input_prompts = []
        self._prompts = [
            # "An astronaut riding a green horse",
            "Lebron James dunking in Mars",
            "Kobe Bryant versus Michael Jordan on the Moon",
            "MS Dhoni hitting a six to the Moon"
        ]
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True, 
            variant="fp16"
        ).to(self._device)
    
    def load_data(self):
        # Prepare batch size number of prompts
        for _ in range(self._batch_size):
            self._input_prompts.append(random.choice((self._prompts)))
    
    def infer(self):
        images = self._model(prompt=self._input_prompts).images
        print('finished an inference!')
        return len(images)

class BertLarge(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._input_prompts = []
        self._prompts = [
            "Lebron James was drafted to [MASK] in 2003",
            "Lebron James won his first championship in [MASK]",
            "Anthony Edwards was picked by the [MASK]",
            "Anthony Edwards is [MASK] years old"
        ]
        self.model_path = "bert-large-uncased"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self._model = BertForMaskedLM.from_pretrained(
            self.model_path
        ).to(self._device)
    
    def load_data(self):
        # Prepare batch size number of prompts
        for _ in range(self._batch_size):
            self._input_prompts.append(random.choice(self._prompts))
        
        # Tokenize
        tokenized_prompts = [self._tokenizer.tokenize(prompt) for prompt in self._input_prompts]

        # Pad
        max_length = max(len(tokens) for tokens in tokenized_prompts)
        padded_tokenized_prompts = [tokens + ["[PAD]"] * (max_length - len(tokens)) for tokens in tokenized_prompts]

        # Convert tokens to input IDs
        input_ids = [self._tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokenized_prompts]
        attention_masks = [[1] * len(tokens) + [0] * (max_length - len(tokens)) for tokens in tokenized_prompts]

        # Convert to PyTorch tensor and move to GPU
        self._input_ids_tensor = torch.tensor(input_ids).to(self._device)
        self._attention_masks_tensors = torch.tensor(attention_masks).to(self._device)

    def infer(self):
        outputs = self._model(self._input_ids_tensor, attention_mask=self._attention_masks_tensors)
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        predicted_tokens = [self._tokenizer.convert_ids_to_tokens(ids) for ids in predicted_token_ids]
        for prompt, pred_tokens in zip(self._input_prompts, predicted_tokens):
            print(f"Prompt: {prompt}")
            print("Predicted tokens:", pred_tokens)
            print()
        return len(predicted_tokens)

class GPT(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._input_prompts = []
        self._prompts = [
            "The NBA season is heating up with intense matchups and standout performances. Predictions for the NBA Finals?",
            "Discuss the impact of recent trades on the competitiveness of teams in the NBA Western Conference.",
            "Analyzing the rise of young talents in the NBA: Who are the top rookies to watch out for this season?",
            "Exploring the debate: Is LeBron James still the most dominant player in the NBA, or are there rising stars challenging his throne?"
        ]
        self.model_path = "EleutherAI/gpt-j-6B"
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self._model = GPTJForCausalLM.from_pretrained(
            self.model_path,
            revision="float16",
            torch_dtype=torch.float16
        ).to(self._device)

    def load_data(self):
        # Prepare batch size number of prompts
        for _ in range(self._batch_size):
            self._input_prompts.append(random.choice(self._prompts))
        
        self._tokenized_prompts = self._tokenizer(self._input_prompts, return_tensors="pt", padding=True, truncation=True)

        # Move inputs to GPU if available
        self._tokenized_prompts.to(self._device)

    def infer(self):
        outputs = self._model.generate(
            **self._tokenized_prompts,
            max_length=100,  
            num_return_sequences=len(self._input_prompts), 
            do_sample=True, 
            temperature=0.7,
            top_k=50,  
            top_p=0.95, 
            pad_token_id=self._tokenizer.eos_token_id  
        )
        # Decode generated responses
        generated_responses = [self._tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Print generated responses
        for i, response in enumerate(generated_responses):
            print(f"Prompt {i+1}: {self._input_prompts[i]}")
            print(f"Generated Response: {response}\n")
        return len(generated_responses)

class Whisper(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        curr_path = pathlib.Path(__file__).parent.resolve()
        self.input_path = os.path.join(curr_path, "data/speech.wav")
        self.model_path = "openai/whisper-small"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = pipeline("automatic-speech-recognition", self.model_path, device=self._device)
    
    def load_data(self):
        self._speeches = []
        for _ in range(self._batch_size):
            audio, _ = torchaudio.load(self.input_path)
            self._speeches.append(audio[0].numpy())

    def infer(self):
        transcriptions = self._model(self._speeches)
        return len(transcriptions)

def get_inference_object(model, device_id, batch_size):
    if model == "diffusion":
        return StableDiffusion(model, device_id, batch_size)
    elif model == "bert":
        return BertLarge(model, device_id, batch_size)
    elif model == "gpt":
        return GPT(model, device_id, batch_size)
    elif model == 'whisper':
        return Whisper(model, device_id, batch_size)
