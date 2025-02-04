class CFG_class():
    def __init__(self):
        self.batch_size = 16
        self.nonGC_lr = 1e-4
        self.GC_lr = 1e-3
        self.num_epochs = 20
        self.eval_epoch = 0
        self.seed = 0
        self.weight_decay = 1e-8
        self.max_len = 256 #Difference in 128 to 256 is around 1GB 
        self.debug = False
        self.name = "exp_test"
        
        self.generation_cfg = {
                "max_length": 40,
                "min_length": 5,
                "num_beams": 1,
                "top_p": 0.9,
                "top_k": 30,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.03,
                "no_repeat_ngram_size": 3,
        }
    
    def log_config(self, log):
        log.info("Config details:")
        d = vars(self)
        for attr in d:
            if type(d[attr]) is dict:
                log.info(f"{attr}:")
                for subattr in d[attr]:
                    log.info(f"  {subattr}: {d[attr][subattr]}")
            else: 
                log.info(f"{attr}: {d[attr]}")
        
        
CFG = CFG_class()
