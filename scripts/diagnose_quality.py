import os
import sys
import torch

def inspect_dataset_file(path):
    print(f"\n=== Inspecting: {path} ===")
    if not os.path.exists(path):
        print("File does not exist!")
        return
    
    with open(path, "rb") as f:
        raw_bytes = f.read(2000)
    
    print("Raw bytes (first 100):", raw_bytes[:100])
    
    try:
        utf8_str = raw_bytes.decode("utf-8")
        print("\nDecoded as utf-8 (first 300 chars):")
        print(repr(utf8_str[:300]))
        print("\nPrint utf-8:")
        print(utf8_str[:300])
        
        # Check if it was double-encoded (e.g. latin-1 / cp1252 to utf-8)
        try:
            fixed_latin = utf8_str.encode("latin-1").decode("utf-8")
            print("\n[SUCCESS] Fixed via encode('latin-1').decode('utf-8'):")
            print(fixed_latin[:300])
        except Exception as e:
            print("\nlatin-1 fix failed:", e)
            
        try:
            fixed_cp1252 = utf8_str.encode("cp1252").decode("utf-8")
            print("\n[SUCCESS] Fixed via encode('cp1252').decode('utf-8'):")
            print(fixed_cp1252[:300])
        except Exception as e:
            print("\ncp1252 fix failed:", e)
            
    except Exception as e:
        print("UTF-8 decode failed:", e)

if __name__ == "__main__":
    inspect_dataset_file("data/uploads/phase2_chat_ro_499c02bfdf759a2e.txt")
    inspect_dataset_file("data/uploads/alpaca_ro_452a2ada90a63301.txt")
    inspect_dataset_file("data/uploads/phase1_pretrain_tech_history_v2_5c3576b51c177271.txt")
