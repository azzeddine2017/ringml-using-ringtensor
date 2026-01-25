load "/universal_cleaner.ring"

func main
    
    oClean = new UniversalCleaner()
    
    # --- Mission 1: Translation (English -> Arabic) ---
    # We define the template here using setTaskWrappers
    
    oClean{
          clear()
          loadFile("downloads/en-ar-small.txt")
          setSeparator("*****") 
          setTaskWrappers("<TO_AR> ", " <SEP> ", " <END>") # The Strategy
          adaptCsv(1, 2, "*****") # Col 1 is En, Col 2 is Ar
          cleanText()
          save("data/ready/train_translation.txt")
          }

    # --- Mission 2: Chat (Alpaca JSONL) ---
    
    oClean{
          clear()
          load_file("downloads/alpaca_data.jsonl")
          setTaskWrappers("<CHAT> ", "\nBot: ", " <END>") 
          adapt_alpaca_jsonl()
          clean_text()
          save("data/ready/train_chat.txt")
          }
    
    see "Data Preparation Complete." + nl