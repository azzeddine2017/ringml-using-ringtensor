# File: src/data/universaldataset.ring
# Description: Professional Data Manager (Load, Clean, Shuffle, Split)
# Author: Azzeddine Remmal


class UniversalDataset
    # Data Storage
    aRawData    = []
    aTrainData  = []
    aTestData   = []
    
    # Configuration
    cFilePath   = ""
    nTestRatio  = 0.2
    bShuffle    = true
    bHasHeader  = false
    
    # State
    nSamples    = 0
    nFeatures   = 0
    nClasses    = 0

    func init cFile
        cFilePath = cFile
        if !fexists(cFilePath) 
            raise("Error: File not found -> " + cFilePath)
        ok

    # --- Configuration Methods (Builder Pattern) ---
    
    func setSplit nRatio
        nTestRatio = nRatio
        return self
        
    func setShuffle bStatus
        bShuffle = bStatus
        return self
        
    func setHeader bStatus
        bHasHeader = bStatus
        return self

    # --- Core Loading Logic ---

    func loadData
        see "Loading dataset: " + cFilePath + "..." + nl
        t1 = clock()
        
        # 1. Read File Content (Fast C-Level Read)
        cContent = read(cFilePath)
        
        # 2. Parse based on Extension
        if right(cFilePath, 4) = ".csv"
            aRawData = CSV2List(cContent)
        elseif right(cFilePath, 5) = ".json"
            aRawData = JSON2List(cContent)
        else
            raise("Unsupported file format. Please use .csv or .json")
        ok
        
        # Free huge string memory immediately
        cContent = "" 
        callgc()
        
        # 3. Remove Header if requested
        if bHasHeader and len(aRawData) > 0
            del(aRawData, 1)
        ok
        
        nSamples = len(aRawData)
        see "Loaded " + nSamples + " samples in " + ((clock()-t1)/clockspersecond()) + "s." + nl
        
        # 4. Perform Split
        processSplit()
        
        return self

    func processSplit
        see "Processing Split (" + ((1-nTestRatio)*100) + "/" + (nTestRatio*100) + ")..." + nl
        
        splitter = new DataSplitter
        sets = splitter.splitData(aRawData, nTestRatio, bShuffle)
        
        aTrainData = sets[1]
        aTestData  = sets[2]
        
        # Clear Raw Data to save memory? 
        # Yes, if we only need train/test.
        aRawData = []
        callgc()
        
        see "Train Size: " + len(aTrainData) + nl
        see "Test Size:  " + len(aTestData) + nl

    # --- Abstract Methods (To be overridden by user) ---
    
    # This method must convert a RAW ROW (List) into [InputTensor, TargetTensor]
    func rowToTensor aRow
        raise("You must define 'rowToTensor' in your subclass!")

    # --- Factory Methods for DataLoaders ---
    
    func getTrainDataset
        return new InternalDatasetAdapter(self, aTrainData)

    func getTestDataset
        return new InternalDatasetAdapter(self, aTestData)

# --- Internal Helper Class ---
# Adapts the raw lists to the RingML Dataset Interface
class InternalDatasetAdapter from Dataset
    oParent # The UniversalDataset instance (to access rowToTensor)
    aList   # The specific list (Train or Test)
    nLen

    func init oParentRef, aDataList
        oParent = oParentRef
        aList   = aDataList
        nLen    = len(aList)

    func length
        return nLen

    func getData nIdx
        # Delegate the specific processing logic back to the parent class
        # This allows the user to define logic once in their subclass
        return oParent.rowToTensor(aList[nIdx])