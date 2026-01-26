# File: main.ring 
# Description: Bi-Directional Transformer Model for Translation
# Author: Azzeddine Remmal

load "AdamModel2.ring"
load "ringml.ring"
load "Inference.ring"
load "BiDirectionalDataset.ring"

see "num cores :" + tensor_get_cores()
tensor_set_threads(2)
setGpuThreshold(5000000)

decimals(8)

func main
    
    train()

func train
    
    # ---------------------------------------------------------
    # 1. Initialization
    # ---------------------------------------------------------
    ? info("[1] Initializing Bi-Directional Pipeline...")

    cVocabPath = "data/vocab.bin"
    cDataPath = "data/en-ar-small.txt"
    if !fexists(cDataPath) raise("Data file not found!") ok
    
    # --- Tokenizer Setup ---
    oTok = new Tokenizer()

    oTok.addToken("<PAD>")   # ID 1 (Padding Token)
    oTok.addToken("<UNK>")   # ID 2 (Unknown Token)
    oTok.addToken("<SEP>")   # ID 3 (Separator Token)
    oTok.addToken("<START>") # ID 4 (Start Token)
    oTok.addToken("<END>")   # ID 5 (End Token)
    oTok.addToken("<TO_AR>") # ID 6 (Task Token)
    oTok.addToken("<TO_EN>") # ID 7 (Task Token)
    
    ? info("    Building Vocabulary...")

    if fexists(cVocabPath)
        oTok.loadVocab(cVocabPath)
    else
        # Read file to build vocab
        cContent = read(cDataPath)
        aLines = str2list(cContent)
        cContent = ""
        aAllText = []
        nLimit = len(aLines)
        if nLimit > 5000 nLimit = 5000 ok
        
        for i = 1 to nLimit
            aParts = split(aLines[i], "*****")
            if len(aParts) >= 2 
                aAllText + aParts[1] + aParts[2]
            ok
        next
        
        oTok.buildVocab(aAllText)
        oTok.saveVocab(cVocabPath)
    ok

    info("    Vocab Size: ")
    see oTok.nVocabSize + nl

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    ? info("[2] Preparing Augmented Data...")
    
    nSeq = 128
    nBatchSize = 64

    # Use custom Bi-Directional Dataset
    oDataset = new BiDirectionalDataset(cDataPath, oTok, nSeq)
    oDataset.setHeader(false)
    oDataset.setShuffle(true)
    oDataset.loadData() 
    
    oTrainLoader = new BatchDataLoader(oDataset.getTrainDataset(), nBatchSize)
    
    # ---------------------------------------------------------
    # 3. Model Configuration
    # ---------------------------------------------------------
    nVocab = oTok.nVocabSize + 1
    nDim   = 128
    nLayers = 2
    nLR = 0.0001
    nWD = 0.0001
    

    oModel = new AdamModel2(nVocab, nSeq, nDim, nLayers)
    //oModel.bWeightTying = true
    //oModel.blocksSummary()
    
    oOptim = new Adam(nLR, nWD) # learning rate=0.0001 , WeightDecay=0.0001
    oLoss  = new CrossEntropyLoss
    
   
    
    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    ? info("[3] Starting Training...")

     oTrainer = new GUI_Trainer_Graph(oModel, oTok, oTrainLoader, oOptim, oLoss, "oTrainer"){
        # Configure Trainer
        
        nEpochs         = 50
        nLogInterval    = 5
        nSaveInterval   = 2

        nSeqLen = nSeq

        # Scheduler & Curriculum
        nBaseLR = 0.001
        nMinLR = 0.00001
        nWarmupSteps = 1000
        nDecayRate = 0.90
        bUseScheduler = true

        # Curriculum
        bUseCurriculum = false 
        
        # Early Stopping
        nBestLoss = 99
        nWorseCount = 0
        nPatienceMax = 30  # Early stopping patience

        cCheckpointDir  = "model/"
        cModelName    = "adamBidirTranslation"
        bMontoring      = true
        //enableCurriculum(32) 
        # RUN
        fit()
    }