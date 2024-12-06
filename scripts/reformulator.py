import copy

from transformers.agents.llm_engine import MessageRole


def prepare_response(original_task, inner_messages, llm_engine):

    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
        }
    ]

    # The first message just repeats the question, so remove it
    # if len(inner_messages) > 1:
    #    del inner_messages[0]

    # copy them to this context
    for message in inner_messages:
        if not message.get("content"):
            continue
        message = copy.deepcopy(message)
        message["role"] = MessageRole.USER
        messages.append(message)

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
        }
    )

    response = llm_engine(messages)

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    print("Reformulated answer is: ", final_answer)

    if "unable to determine" in final_answer.lower():
        messages.append({"role": MessageRole.ASSISTANT, "content": response})
        messages.append(
            {
                "role": MessageRole.USER,
                "content": """
I understand that a definitive answer could not be determined. Please make a well-informed EDUCATED GUESS based on the conversation.

To output the educated guess, use the following template: EDUCATED GUESS: [YOUR EDUCATED GUESS]
Your EDUCATED GUESS should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
ADDITIONALLY, your EDUCATED GUESS MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip(),
            }
        )

        response = llm_engine(messages)
        print("\n>>>Making an educated guess.\n", response)
        final_answer = response.split("EDUCATED GUESS: ")[-1].strip()
    return final_answer
