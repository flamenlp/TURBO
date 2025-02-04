#Importing Libraries
import os
import time
import inspect
import sys
from tqdm import tqdm
import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers.optimization import get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import TURBO
from torchvision import transforms
import model as model_file
from torch.utils.data import DataLoader
from dataset import MOREPlusDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from cfg import CFG
from transformers import BartTokenizer
import logging
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from rouge_score import rouge_scorer
##################################################

#Function to set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

#Function to send data to GPU
def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

#Functions to evaluate text similarity metrics
def eval_metrics(gen_text_file, ref_text_file):
    # calculate metrices
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_list = f.readlines()
    with open(gen_text_file, 'r', encoding='utf-8') as f:
        gen_list = f.readlines()

    ref_dict = {}
    gen_dict = {}

    for i in range(len(gen_list)):
        ref = ' '.join(nltk.word_tokenize(ref_list[i].lower()))
        gen = ' '.join(nltk.word_tokenize(gen_list[i].lower()))

        ref_dict[i] = [ref]
        gen_dict[i] = [gen]

    scores = eval_metrics_helper(gen_dict, ref_dict)

    for k in scores:
        scores[k] = f"{scores[k] * 100:.3f}"
    return scores

def eval_metrics_helper(gen_dict, ref_dict):
    gen_list=[]
    ref_list=[]
    print("Hypothesis list:")
    for i in gen_dict.values():
        gen_list.append(i)
        print(i, end=" ")
    for i in ref_dict.values():
        ref_list.append(i)
    
    scores_dict = {}
    
    b = Bleu()
    score, _ = b.compute_score(gts=ref_dict, res=gen_dict)
    b1, b2, b3, b4 = score

    r = Rouge()
    score, _ = r.compute_score(gts=ref_dict, res=gen_dict)

    rl = score
    count=0
    rouge1=0
    rouge2=0
    meteor=0
    for reference, hypothesis in zip(ref_list, gen_list):
        reference=str(reference).strip('[]\'')
        hypothesis=str(hypothesis).strip('[]\'')

        count += 1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        meteor += nltk.translate.meteor_score.meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(hypothesis))
        scores = scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
    rouge1 = rouge1 / count
    rouge2 = rouge2 / count
    meteor = meteor / count


    scores_dict['Bleu_1'] = b1
    scores_dict['Bleu_2'] = b2
    scores_dict['Bleu_3'] = b3
    scores_dict['Bleu_4'] = b4
    scores_dict['Rouge_L'] = rl
    scores_dict['Rouge1'] = rouge1
    scores_dict['Rouge2'] = rouge2
    scores_dict['METEOR'] = meteor
    return scores_dict

#Function to create and return dataloaders
def load_data(tkr, CFG):    
    
    #Defining transform to apply to images
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #Train
    train_dataset = GoldDataset (
        data_file= 'Dataset/train_df.tsv', 
        CC_file = 'Dataset/CC_train.pkl', 
        D_file='Dataset/D_train.pkl', 
        DC_file = 'Dataset/DC_train.pkl', 
        O_file = 'Dataset/O_train.pkl', 
        OC_file = 'Dataset/OC_train.pkl', 
        adj_file = 'Dataset/KG_train.pkl', 
        path_to_images = 'Dataset/images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        CFG=CFG
    )
    
    #Val
    val_dataset = GoldDataset (
        data_file= 'Dataset/val_df.tsv', 
        CC_file = 'Dataset/CC_val.pkl', 
        D_file='Dataset/D_val.pkl', 
        DC_file = 'Dataset/DC_val.pkl', 
        O_file = 'Dataset/O_val.pkl', 
        OC_file = 'Dataset/OC_val.pkl', 
        adj_file = 'Dataset/KG_val.pkl', 
        path_to_images = 'Dataset/images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        CFG=CFG
    )
    
    #Test
    test_dataset = GoldDataset (
        data_file= 'Dataset/test_df.tsv', 
        CC_file = 'Dataset/CC_test.pkl', 
        D_file='Dataset/D_test.pkl', 
        DC_file = 'Dataset/DC_test.pkl', 
        O_file = 'Dataset/O_test.pkl', 
        OC_file = 'Dataset/OC_test.pkl', 
        adj_file = 'Dataset/KG_test.pkl', 
        path_to_images = 'Dataset/images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        CFG=CFG
    )

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader

#Function to run for validation loop
def eval_net(model, loader, device):
    ppl_mean = 0
    model.eval() #Only inference

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            ppl = model(**batch,mode='eval')
            ppl_mean += ppl.cpu().numpy()

    ppl_mean = ppl_mean / idx 

    return ppl_mean

#Function to generate explanations on test set
def gen_net(ep, model, loader, device):
    
    #All explanations are written (in order) to gen_file_name
    gen_file_name = f'Generations/gen_{ep}.txt'
    gen_file = open(gen_file_name, 'w', encoding='utf-8') 
    
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            _,query_infer = model(**batch,mode='gen')
            gen_file.write('\n'.join(query_infer) + '\n')

    gen_file.flush()
    gen_file.close()
    
    #Evaluate metrics on generated explanations
    scores = eval_metrics(gen_file_name, 'ref_explanation_out.txt')

    return scores


def load_model(model, epoch):
    state_dict = torch.load('Models/epoch' + str(epoch).zfill(3) + '.pth')
    model.load_state_dict(state_dict)
    return model


def save_model(model, epoch):
    torch.save(model.state_dict(), 'Models/epoch' + str(epoch).zfill(3) + '.pth')

def run_stage(CFG, model, lr_sche, opt,
              train_loader, eval_loader, test_loader,
              device, log):
    print_every = 2
    max_epoch = int(CFG.max_epoch)

    scores = []
    imp_vals = []
    for epoch in range(max_epoch):
        model.train()
        log.info(f"{'-' * 20} Current Epoch:  {epoch} {'-' * 20}")

        time_now = time.time()
        show_loss = 0

        for idx, batch in enumerate(train_loader):
            opt.zero_grad()

            batch = send_to_device(batch, device)
            loss = model(**batch)

            loss.backward()
            opt.step()

            show_loss += loss.detach().cpu().numpy()
            
            #Print stats
            if idx % print_every == print_every - 1:
                cost_time = time.time() - time_now
                time_now = time.time()
                log.info(
                    f'step: {idx + 1}/{len(train_loader) + 1} | time cost {cost_time:.2f}s | loss: {(show_loss / print_every):.4f}')
                show_loss = 0

            lr_sche.step()

        #Logging current learning rates
        for _, param_grp in enumerate(opt.param_groups):
            log.info(f"[Param group {param_grp['name']}]: Current lr is {param_grp['lr']}")

        #Validation
        log.info('Validation Started')
        ppl = eval_net(model, eval_loader, device)

        #Save checkpoint
        save_model(model, epoch)
        log.info('Model Saved!')

        #Load saved checkpoint 
        model = load_model(model, epoch)
        log.info(f'Model Loaded!')
        
        ppl = eval_net(model, test_loader, device)
        score = gen_net(epoch, model, test_loader, device)
        score['PPL'] = ppl
        print(score)
        log.info(f'Scores for epoch {epoch}: {score}')
        scores.append(score)

        imp_vals.append({})
        for _, param_grp in enumerate(opt.param_groups):
            imp_vals[-1][f"lr_{param_grp['name']}"] = param_grp['lr']

    return scores, imp_vals

#  @hydra.main(config_path="conf", config_name="cfg",version_base='1.2.0')
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(int(CFG.seed))
   
    #Set up logging
    logging.basicConfig(filename='details.log', filemode = "w",level=logging.DEBUG,
                        format='%(levelname)s: %(message)s')
    global log
    log = logging.getLogger()

    tkr = BartTokenizer.from_pretrained("facebook/bart-base")
    model = TURBO(CFG, tkr).to(device)
    
    train_dataloader, eval_dataloader, test_dataloader = load_data(tkr, CFG)
    
    from_scratch_params = list(map(id, nn.ModuleList([model.gc]).parameters()))
    other_params = filter(lambda p: id(p) not in from_scratch_params, model.parameters())
    
    optimizer = AdamW([
        {'params': other_params, 'lr': CFG.nonGC_lr, 'name': 'NonGC'},
        {'params': nn.ModuleList([model.gc]).parameters(), 'lr': CFG.GC_lr, 'name': 'GC'}],
        weight_decay=CFG.weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=int(CFG.max_epoch) * len(train_dataloader))
    
    #Log source code of model file
    log.info(inspect.getsource(model_file))

    log.info("Starting training...")

    #Log device
    log.info(f'Found device: {device}')
    
    #Log config
    CFG.log_config(log)

    #Log sizes of dataloaders
    log.info(f"train data: {CFG.batch_size * len(train_dataloader)}")
    log.info(f"eval data: {CFG.batch_size * len(eval_dataloader)}")
    log.info(f"test data: {CFG.batch_size * len(test_dataloader)}")

    scores, imp_vals = run_stage(CFG=CFG, model=model,lr_sche=lr_scheduler, opt=optimizer,
                       train_loader=train_dataloader, eval_loader=eval_dataloader, 
                       test_loader=test_dataloader,device=device, log=log)

    #Logging all scores and other important values for reference
    for idx, score in enumerate(scores):
        log.info(f"{'-' * 20} Epoch {idx} {'-' * 20}")
        
        for key in score.keys():
            log.info(f'{key}: {score[key]}')
        
        for key in imp_vals[idx].keys():
            log.info(f'{key}: {imp_vals[idx][key]}')


if __name__ == '__main__':
    main()
