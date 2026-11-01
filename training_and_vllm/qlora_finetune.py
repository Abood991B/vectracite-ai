"""
QLoRA (4-bit NF4) Fine-Tuning Pipeline for Financial Citation & Structured JSON Extraction.
Trains Llama-3.1-8B-Instruct on PatronusAI/financebench ground-truth context-question-answer triples
using Hugging Face TRL (SFTTrainer), PEFT (LoRA r=16, alpha=32), and BitsAndBytes 4-bit quantization.
"""

import argparse
import logging
from typing import Dict
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FINANCEBENCH_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are VectraCite-AI, a financial analyst model trained for strict citation-grounded extraction.
Extract the exact answer from the financial filing evidence and cite [DOC_NAME, Page X].<|eot_id|>
<|start_header_id|>user<|end_header_id|>
DOCUMENT: {doc_name} (Page {page_number})
EVIDENCE:
{evidence}

QUESTION:
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer} [Citation: {doc_name}, Page {page_number}]<|eot_id|>"""


def format_financebench_row(row: Dict) -> Dict[str, str]:
    text = FINANCEBENCH_PROMPT_TEMPLATE.format(
        doc_name=row.get("doc_name", "SEC_FILING"),
        page_number=row.get("page_number", 1),
        evidence=row.get("evidence", ""),
        question=row.get("question", ""),
        answer=row.get("answer", ""),
    )
    return {"text": text}


def run_qlora_training(
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "./qlora_checkpoints/vectracite-llama3.1-8b-lora",
    num_epochs: int = 3,
    dry_run: bool = False,
) -> None:
    logger.info("Loading PatronusAI/financebench training split...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    formatted_ds = dataset.map(format_financebench_row)
    logger.info("Prepared %d formatted instruction-tuning examples.", len(formatted_ds))

    if dry_run:
        logger.info("Dry-run sample preview:\n%s", formatted_ds[0]["text"][:500])
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="epoch",
        report_to="mlflow",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_ds,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved QLoRA adapter weights to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset formatting without GPU")
    args = parser.parse_args()
    run_qlora_training(dry_run=args.dry_run)
