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
    bShuffle    = false
    bHasHeader  = false
    bSplit      = false
    bDesply     = false
    
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
    
    func setRatio nRatio
        nTestRatio = nRatio
        return self
        
    func setShuffle bStatus
        bShuffle = bStatus
        return self
        
    func setHeader bStatus
        bHasHeader = bStatus
        return self

    func setSplit bStatus
        bSplit = bStatus
        return self
    
    # --- Core Loading Logic ---

    func loadData
        if bDesply 
            oStyl.cyan(:NONE, "Loading dataset: ") 
            oStyl.green(:NONE, cFilePath)
            oStyl.cyan(:NONE, " ..."+ nl)
        ok
        
        t1 = clock()
        
        # 1. Read File Content (Fast C-Level Read)
        cContent = read(cFilePath)
        
        # 2. Parse based on Extension
        if right(cFilePath, 4) = ".csv"
            aRawData = CSV2List(cContent)
        elseif right(cFilePath, 5) = ".json"
            aRawData = JSON2List(cContent)
        elseif right(cFilePath, 4) = ".txt"
            aRawData = str2List(cContent)
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
        if bDesply 
            oStyl.cyan(:NONE, "Loaded " ) 
            oStyl.green(:NONE, nSamples ) 
            oStyl.cyan(:NONE, " samples in " )
            oStyl.green(:NONE, "" + ((clock()-t1)/clockspersecond()))
            oStyl.cyan(:NONE, "s."+ nl)
        ok
            
        # 4. Perform Split
        if bSplit 
            processSplit()
        else
            aTrainData = aRawData
            aTestData  = []
        ok
        
        return self

    # Function to sort data by length (shortest first)
    func sortDataByLength
        ? info("    Sorting Dataset by Length (Smart Batching Strategy)...")
        
        # 1. Create a temporary list [length, original line]
        aTemp = []
        for i=1 to len(aTrainData)
            cLine = aTrainData[i]
            # the length of the line
            nLen = len(cLine) 
            aTemp + [nLen, cLine]
        next
        
        # 2. sort by length
        aTemp = sort(aTemp, 1)
        
        # 3. update the train data
        aTrainData = []
        for item in aTemp
            aTrainData + item[2] # the original line
        next
        
        if bDesply
            ? oStyl.cyan(:BOLD, "  Dataset Sorted. Shortest samples come first.")
            oStyl.cyan(:BOLD, "first sample: (")
            oStyl.green(:BOLD, aTrainData[1])
            ? oStyl.cyan(:BOLD, ")")
            oStyl.cyan(:BOLD, "last sample: (")
            oStyl.green(:BOLD, aTrainData[len(aTrainData)])
            ? oStyl.cyan(:BOLD, ")")
        ok  
    
    func processSplit
        if bDesply
            oStyl.cyan(:NONE, "Processing Split ( ")
            oStyl.green(:NONE, "" + ((1-nTestRatio)*100))
            oStyl.cyan(:NONE, " / ") 
            oStyl.green(:NONE, "" + (nTestRatio*100)) 
            oStyl.cyan(:NONE, " ) ..."+ nl)
        ok
        
        splitter = new DataSplitter
        sets = splitter.splitData(aRawData, nTestRatio, bShuffle)
        
        aTrainData = sets[1]
        aTestData  = sets[2]
        
        # Clear Raw Data to save memory? 
        # Yes, if we only need train/test.
        aRawData = []
        callgc()
        
        if bDesply
            oStyl.cyan(:NONE, "Train Size: ")
            oStyl.green(:NONE, "" + len(aTrainData))
            oStyl.cyan(:NONE, "Test Size:  ")
            oStyl.green(:NONE, "" + len(aTestData) + nl)
        ok       

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