import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def main():
    
    model_name = "kakaocorp/kanana-nano-2.1b-base"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # <|end_of_text|> 128001

    qna_list = []
    with open("/media/ct/disk2/project/jeon/study/LLM/data/jeoncustomdata.txt", "r") as file:
        for line in file:
            qna = line.strip().split('|')  # 안내: 입력 문서의 '|'는 질문과 답변을 구분하는 문자
            input_str = qna[0] + " " + qna[1]
            item = {'q': qna[0], 'input': input_str, 'q_ids': tokenizer.encode(qna[0]), 'input_ids': tokenizer.encode(input_str)}
            qna_list.append(item)

    max_length = max(len(item['input_ids']) for item in qna_list)

    print(qna_list)
    print(max_length)

    # 파인튜닝 전에 어떻게 응답하는지 확인
    questions = [qna['q'] for qna in qna_list]
    questions.append("너에 대해서 설명해봐.")
    questions.append("이처럼 인간처럼 생각하고 행동하는 AI 모델은 ")
    questions.append("인공지능의 장점은")
    questions.append("전사영에 대해서 얘기해봐.")

    input_ids = tokenizer(
        questions,
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
        )

    output_list = output.tolist()

    for i, output in enumerate(output_list):
        print(f"Q{i}: {tokenizer.decode(output, skip_special_tokens=True)}")
        
    EOT = tokenizer.eos_token_id  # 128001

    class MyDataset(Dataset):
        def __init__(self, qna_list, max_length, tokenizer):
            self.input_ids = []
            self.target_ids = []

            for qa in qna_list:
                token_ids = qa['input_ids']
                prompt_ids = tokenizer.encode(qa['q'] + " ")  # 정확한 prompt 길이 계산
                len_prompt = len(prompt_ids)
                
                input_chunk = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))
                target_chunk = token_ids[1:] + [-100] * (max_length - len(token_ids[1:]))
                
                # prompt 부분 ignore: target[:len_prompt - 1] = -100
                if len_prompt - 1 > 0:
                    target_chunk[:len_prompt - 1] = [-100] * (len_prompt - 1)

                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    dataset = MyDataset(qna_list, max_length=max_length, tokenizer=tokenizer)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

    i = iter(train_loader)
    x, y = next(i)

    y_temp = y[0].tolist()
    y_temp = [val for val in y_temp if val != -100]  # -100은 제외하고 디코딩

    print(tokenizer.decode(x[0].tolist()))
    print(tokenizer.decode(y_temp))

    torch.manual_seed(123)
    optimizer = AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    tokens_seen, global_step = 0, -1

    losses = []

    for epoch in range(10):
        model.train()  # Set model to training mode
        
        epoch_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            logits = model(input_batch).logits  # 뒤에 .logits를 붙여서 tensor만 가져옴

            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            accelerator.backward(loss)  # Use accelerator for backward
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            print(f"{global_step} Tokens seen: {tokens_seen}")

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch: {epoch}, Loss: {avg_loss}")
        torch.save(accelerator.unwrap_model(model).state_dict(), "model_" + str(epoch).zfill(3) + ".pth")
        
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # 파인튜닝 후에 어떻게 응답하는지 확인
    accelerator.unwrap_model(model).load_state_dict(torch.load("model_009.pth", map_location=accelerator.device, weights_only=True))
    model.eval()

    questions = [qna['q'] for qna in qna_list]
    questions.append("전사영이 매일하는 게임은?") 
    questions.append("전사영에 대해서 얘기해봐.")
    questions.append("카나나 모델에 대해서 설명해봐.")
    questions.append("이처럼 인간처럼 생각하고 행동하는 AI 모델은 ")
    questions.append("인공지능의 장점은")

    for i, q in enumerate(questions):
        input_ids = tokenizer(
            q,
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=32,
                attention_mask=(input_ids != tokenizer.pad_token_id).long(),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        
    # Interactive part
    user_input = input()
    input_ids = tokenizer(
        user_input,
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    print(f"Q: {tokenizer.decode(output[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()