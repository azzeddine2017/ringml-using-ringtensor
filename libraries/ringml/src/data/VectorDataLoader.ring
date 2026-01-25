# File: src/data/VectorDataLoader.ring

class VectorDataLoader from BatchDataLoader

    oVDB        # The vector database
    cTaskTag    # The tag (like "[TRSL]")
    oDataLogic  # Object containing the rowToTensor logic

    func init oVectorDB, oLogicRef, cTag, nBatch
        oVDB = oVectorDB
        oDataLogic = oLogicRef
        cTaskTag = cTag
        nBatchSize = nBatch
        
        # 1. Filtering data from memory based on the tag
        aFilteredList = []
        for cRecord in oVDB.aMetadata
            # Searching for the tag at the start of the metadata
            if left(cRecord, len(cTaskTag)) = cTaskTag
                # Extracting the clean text (after the tag and the :: separator)
                nStart = len(cTaskTag) + 3 # +3 for the separator ":: "
                aFilteredList + substr(cRecord, nStart)
            ok
        next

        if len(aFilteredList) = 0
            raise("VectorDataLoader: No data found for tag " + cTaskTag)
        ok

        # 2. Creating the Adapter to connect the list to the conversion logic
        oMemoryDataset = new InternalDatasetAdapter(oDataLogic, aFilteredList)
        
        # 3. Initializing the original DataLoader using the memory
        
        return self