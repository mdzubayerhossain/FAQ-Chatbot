import json

# Input and output file paths
input_file = r"D:\Coding\Llama chatbot\FAQ.txt"
output_file = "faq_data.json"

# Initialize an empty list to store FAQ entries
faq_data = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_entry = {}
for line in lines:
    line = line.strip()  # Remove leading/trailing spaces
    if line.startswith("1.") or line.startswith("2."):  # Start of a new question
        if current_entry:  # Save the previous entry
            faq_data.append(current_entry)
        current_entry = {"question": line[3:].strip(), "scenarios": []}
    elif line.startswith("Probing Question:"):
        current_entry["probing_question"] = line.split("Probing Question:")[1].strip()
    elif line.startswith("Senario"):
        scenario = {"condition": line.split(":")[1].strip(), "answer": ""}
        current_entry["scenarios"].append(scenario)
    elif line.startswith("ANS:"):
        if current_entry["scenarios"]:
            current_entry["scenarios"][-1]["answer"] = line.split("ANS:")[1].strip()
        else:
            current_entry["answer"] = line.split("ANS:")[1].strip()
    elif line.startswith("(("):  # Notes for the agent
        current_entry["agent_notes"] = line.strip("()")
        
# Append the last entry
if current_entry:
    faq_data.append(current_entry)

# Save as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=4)

print(f"FAQ data successfully converted to {output_file}")
