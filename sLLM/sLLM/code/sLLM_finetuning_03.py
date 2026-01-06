### instruct 모델 훈련 ### 

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct" # "-instruct" 지시에 따르도록 파인튜닝(사후훈련)이 된 모델
# model_name = "kakaocorp/kanana-nano-2.1b-base" # base 모델로도 지시 훈련이 됩니다.
# model_name = "microsoft/Phi-4-mini-instruct" # MIT 라이센스라서 상업적 사용 가능, 아래에서 epoch 50번 정도면 훈련 됩니다.

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

messages = [
    {'role': 'system', 'content': 'You are a helpful AI assistant developed by Kakao'},
    {'role': 'user', 'content': '전사영이 좋아하는 게임은?'},
    {'role': 'assistant', 'content': '전사영은 피파온라인을 좋아해서 자주 합니다.'}
]

tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

print(tokens)
# print(tokenizer.decode(tokens))



qna_list = []
with open("C:/Users/jsy/Desktop/coretech/sLLM/sLLM/data/jeoncustomdata.txt", "r", encoding='utf-8') as file:
    for line in file:
        qna = line.strip().split('|') # 안내: 입력 문서의 '|'는 질문과 답변을 구분하는 문자
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."}, # 모든 질문 공통
            {"role": "user", "content": qna[0]},     # 질문 부분
            {"role": "assistant", "content": qna[1]} # 답변 부분
        ]
        q = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        input_str = tokenizer.apply_chat_template(messages[:3], tokenize=False, add_generation_prompt=True)
        # print(q)
        # print(input_str)
        input_str = input_str[:-len('start_header_id\>assistant<|end_header_id|>')-4]
        print(input_str)
        # print("--------------")
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        input_ids = tokenizer.encode(input_str, add_special_tokens=False)
        qna_list.append({'q':q, 'input':input_str, 'q_ids':q_ids, 'input_ids':input_ids})

max_length = max(len(i['input_ids']) for i in qna_list)

print(qna_list)
print(max_length) # 토큰화 후에 가장 긴 길이 (패딩으로 채우기 위함)

import torch
from torch.utils.data import Dataset, DataLoader

EOT = 128009

class MyDataset(Dataset):
    def __init__(self, qna_list, max_length):
        self.input_ids = []
        self.target_ids = []

        # token_ids = tokenizer.encode("<|endoftext|>" + txt, allowed_special={"<|endoftext|>"})
        for qa in qna_list:
            token_ids = qa['input_ids']
            input_chunk = token_ids
            target_chunk = token_ids[1:]
            input_chunk += [EOT]* (max_length - len(input_chunk))
            target_chunk +=  [EOT]* (max_length - len(target_chunk))
            len_ignore = len(qa['q_ids']) - 1 # target은 한 글자가 짧기 때문
            target_chunk[:len_ignore] = [-100] * len_ignore 

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

dataset = MyDataset(qna_list, max_length=max_length)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

i = iter(train_loader)
x, y = next(i)

y_temp = y[0].tolist()
y_temp = [x for x in y_temp if x != -100] # -100은 제외하고 디코딩

print(tokenizer.decode(x[0].tolist()))
print("----------")
print(tokenizer.decode(y_temp))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
torch.manual_seed(123)
model.to(device)

# 파인튜닝 전에 어떻게 대답하는지 확인
questions = [ qna['q_ids'] for qna in qna_list]

for i, q_ids in enumerate(questions):

    model.eval()
    with torch.no_grad():
        output = model.generate(
            torch.tensor([q_ids]).to("cuda"),
            max_new_tokens=32,
            #attention_mask = (input_ids != 0).long(),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # temperature=1.2,
            # top_k=5
        )

    output_list = output.tolist()

    print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")

tokens_seen, global_step = 0, -1

losses = []

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)

# for epoch in range(10):
#     model.train()  # Set model to training mode
    
#     epoch_loss = 0
#     for input_batch, target_batch in train_loader:
#         optimizer.zero_grad() # Reset loss gradients from previous batch iteration
#         input_batch, target_batch = input_batch.to(device), target_batch.to(device)

#         logits = model(input_batch).logits # 뒤에 .logits를 붙여서 tensor만 가져옴

#         loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#         epoch_loss += loss.item()
#         loss.backward() # Calculate loss gradients
#         optimizer.step() # Update model weights using loss gradients
#         tokens_seen += input_batch.numel()
#         global_step += 1

#         print(f"{global_step} Tokens seen: {tokens_seen}")

#         # if global_step % 1000 == 0:
#         #     print(f"Tokens seen: {tokens_seen}")
#         # Optional evaluation step

#     avg_loss = epoch_loss / len(train_loader)
#     losses.append(avg_loss)
#     print(f"Epoch: {epoch}, Loss: {avg_loss}")
#     torch.save(model.state_dict(), "model_" + str(epoch).zfill(3) + ".pth")

    # num_batches = 8 에서 4분 22초 (결과가 안좋음)
    # num_batches = 4 에서 6분 30초 (결과가 안좋음)
    # num_batches = 2 에서 4분 37초 (결과 정확)

# 파인튜닝 후에 어떻게 응답하는지 확인

model.load_state_dict(torch.load("model_009.pth", map_location=device, weights_only=True))
model.eval()

# import matplotlib.pyplot as plt

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.show()

messages = [
            {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."}, # 모든 질문 공통
            {"role": "user", "content": input()},     # 질문 부분
        ]

model.eval()
with torch.no_grad():
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    output = model.generate(
        torch.tensor([ids]).to("cuda"),
        max_new_tokens=64,
        #attention_mask = (input_ids != 0).long(),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # temperature=1.2,
        # top_k=5
    )

output_list = output.tolist()

print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")     