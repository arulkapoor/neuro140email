
import openai
import pickle
openai.api_key = "<>"

prompt = """
Imagine you are an email data generation tool for fine tuning a model.
Example 1: PROMPT: Given the inputs
"{
VC Name: Nabeel Hyatt,
Team: Gavin Uberti, Chris Zhu,
Company: We speed up LLM performance through ASICs and have some exciting technical progress.
}"

You produce the following introduction email after using LinkedIn to find information on 'VC Name' and 'Team'

COMPLETION:
"
Hello!

For Nabeel, Gavin and Chris are Harvard students in Prod working on speeding up LLM performance through ASICs and have had some exciting technical progress. They’ve gotten a couple of term sheets and are thinking of opening up their round.

For Gavin and Chris, Nabeel is a GP at Spark and has an insane investing track record. Nabeel has been super helpful in supporting Prod this year and figured you all would benefit from meeting.

Hope you find time to connect!

Best,
Arul
"

Example 2: PROMPT: Given the inputs
"{
VC Name: Kanyi Maqubela,
Team: Ethan Thornton,
Company: We build projectiles that can travel significantly farther and faster than traditional ammunition, but at the same cost (eventually even less expensive with economies of scale).
}"

You produce the following introduction email after using LinkedIn to find information on 'VC Name' and 'Team'

COMPLETION:
"
Hello!

Kanyi, Ethan is an MIT dropout working on a super cool defense tech startup who we've been working with through Prod. He's thinking about raising a round in the next few weeks so figured you may be interested in meeting.

For Ethan, Kanyi is the Managing Partner at Kindred and helped give some advice to Prod as we were navigating our own nonprofit sponsorship raise!

Hope you find time to chat!

Best,
Arul
"
Example 3:

"""

x = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [{"role": "user", "content": prompt},{"role": "user", "content": "generate another example"}],
    max_tokens=256,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
fine_tune_data = []
for i in range(40):
    y = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt},{"role": "user", "content": "generate another example"}],
        max_tokens=256,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    fine_tune_data.append(y["choices"][0]['message']['content'])
    print(y["choices"][0]['message']['content'])


pickle.dump(fine_tune_data, open( "fine_tune3.p", "wb" ) )