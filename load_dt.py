from datasets import load_dataset

# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf")
train_chosens = dataset['train']['chosen']

#Role = Literal["system", "user", "assistant"]
def hh_stri_2_json(data):
  data_sps = data.split()
  data_sps = [d for d in data_sps if d.strip()]

  context = ''
  role = ''
  rss = []
  for d in data_sps:
    if d =='Human:':
      if context:
        rss.append({'role':role,'content':context})
        context = ''
      role = 'user'
    elif d == 'Assistant:':
      if context:
        rss.append({'role':role,'content':context})
        context = ''
      role = 'assistant'
    else:
      context += ' ' + d

  if context:
    rss.append({'role':role,'content':context})

  return rss

from transformers import AutoTokenizer
PATH_TO_CONVERTED_TOKENIZER = 'NousResearch/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER,add_eos_token=False,add_bos_token=False)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def make_input(data):
  #  29914, 25580
  prompt_bos = '[INST]'
  prompt_eos = '[/INST]'
  epoch_begin = '<s>' # 1
  epoch_end = '</s>' # 2
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

 # conversation_prompt = f"{epoch_begin}"
  leg = len(data)
  if leg%2 != 0:
    return ''

  stri = ''
  sys_prompt = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

  for i in range(0,leg,2):
    prompt = data[i]['content']
    ans = data[i+1]['content']
    if i == 0:
      stri += f"{epoch_begin} {prompt_bos} {sys_prompt} {prompt} {prompt_eos} {ans} {epoch_end}"
    else:
      stri += f"{epoch_begin} {prompt_bos} {prompt} {prompt_eos} {ans} {epoch_end}"

  return stri

def make_tensor(tokenizer,stris,max_length=2048):
  if type(stris) != list:
    stris = [stris]

  pts = tokenizer(stris,padding=True,return_tensors='pt')
  labels = pts['input_ids'].clone()
  labels_list = pts['input_ids'].clone().tolist()

  for i,label in enumerate(labels_list):
  #  leg = len(label)
    lasts = []
    conversation_lasts = []

    for j,_label in enumerate(label):
      if _label == 2: # 
        lasts.append(j)

      if _label == 25580:
        conversation_lasts.append(j+2)


    #last = leg - label[::-1].index(1) - 1 + 1
    #conversation_last = leg - label[::-1].index(25580) - 1 + 2

    if conversation_lasts[-1]>=lasts[-1]:
      raise Exception('worong position of conversation_last and last')
    #print(conversation_lasts[-1],lasts[-1])
    labels[i][:conversation_lasts[-1]] = -100
    labels[i][lasts[-1]+1:] = -100


  pts['labels'] = labels
  return pts


#from transformers.models.flaubert import configuration_flaubert


format_sts = []
for chosen in train_chosens:
  dic = hh_stri_2_json(chosen)
  if not dic:
    continue

  format_stri = make_input(dic)
  if not format_stri:
    continue
  format_sts.append(format_stri)
  
  
def collan(datas):
  return make_tensor(tokenizer,datas)


from torch.utils.data import DataLoader
#DataLoader(training_data, batch_size=64, shuffle=True)
dt = DataLoader(format_sts,batch_size=4,collate_fn=collan)

# sanity check

#for d in dt:
  #pass

