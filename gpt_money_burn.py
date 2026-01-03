import openai
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor

# Your OpenAI API key
api_key = ""

# Get the encoding for a specific model
enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

# Cost per incoming token (in USD)
cost_per_incoming_token = 0.003 / 1000

# Cost per response token (in USD)
cost_per_response_token = 0.004 / 1000

# Starting money spent (in USD)
total_cost = 4.49

# Configure OpenAI with your API key
openai.api_key = api_key

# Create a prompt for ChatGPT with maximum tokens
prompt = "Please provide a comprehensive and detailed overview of the history of human civilization. Start from the " \
         "earliest known human settlements, detailing the discovery of fire, invention of tools, formation of tribes, " \
         "early communities, and the emergence of language and communication. Explore the development of agriculture, " \
         "the domestication of animals, the rise of trade routes, and the establishment of the first cities and " \
         "governments. Discuss the rise and fall of ancient empires such as Mesopotamia, Egypt, Greece, Rome, China, " \
         "and the Indus Valley, including their contributions to art, literature, philosophy, science, governance, " \
         "architecture, military strategies, and the influence of religion and mythology. Examine the Middle Ages, " \
         "focusing on feudalism, the church's role in society, the Crusades, the Black Death, the emergence of " \
         "universities, scholasticism, chivalry, and the development of Gothic architecture. Delve into the " \
         "Renaissance, highlighting the revival of classical learning, humanism, the invention of the printing press, " \
         "the works of Leonardo da Vinci, Michelangelo, and other artists, the exploration of new scientific ideas " \
         "and methodologies, and the patronage of arts by powerful families. Explore the Age of Exploration, " \
         "the colonization of the Americas, the impact of European expansion, the slave trade, the cultural exchange " \
         "between the Old and New Worlds, and the technological advancements in navigation and cartography. Discuss " \
         "the Scientific Revolution, the Enlightenment, the contributions of figures like Galileo, Newton, Voltaire, " \
         "the development of political philosophies, revolutions, the rise of democratic ideals, and the separation " \
         "of church and state. Analyze the Industrial Revolution, including technological advancements, urbanization, " \
         "social changes, the growth of capitalism, labor movements, the emergence of new political ideologies, " \
         "the impact on women's rights, and the challenges of urban poverty and public health. Examine the World " \
         "Wars, the Cold War, the rise of democracy, the struggle for human rights, the formation of international " \
         "organizations, the challenges of nationalism, decolonization, the development of nuclear technology, " \
         "and the ideological conflicts between capitalism and communism. Discuss modern developments such as the " \
         "internet, globalization, environmental challenges, medical advancements, space exploration, artificial " \
         "intelligence, the ongoing pursuit of peace and prosperity, the role of media and information technology, " \
         "the challenges of cybersecurity, and the ethical considerations of biotechnology and genetic engineering. " \
         "Include details on significant events, key figures, cultural developments, technological advancements, " \
         "and the impact on various regions around the world. Provide insights into the interconnectedness of these " \
         "events and how they have shaped the current state of human civilization. Consider the philosophical " \
         "implications of these historical developments and reflect on what they reveal about human nature, ethics, " \
         "morality, religion, art, literature, music, and the pursuit of knowledge and wisdom. Finally, speculate on " \
         "the future of humanity, considering potential advancements in technology, social progress, " \
         "global challenges such as climate change and inequality, ethical dilemmas related to genetic engineering " \
         "and automation, and the potential for human exploration and colonization of other planets. Reflect on the " \
         "lessons of history and how they might guide humanity in facing these future challenges and opportunities, " \
         "the role of education and lifelong learning, the potential for global collaboration and conflict " \
         "resolution, and the importance of empathy, compassion, creativity, and innovation in shaping a positive " \
         "future for all of humanity."


# Function to make the API call
def make_call():
    global total_cost
    incoming_tokens = len(enc.encode(prompt))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": prompt}]
    )
    response_tokens = response['usage']['total_tokens'] - incoming_tokens
    cost = (incoming_tokens * cost_per_incoming_token) + (response_tokens * cost_per_response_token)
    total_cost += cost
    print(f"Incoming tokens: {incoming_tokens}\nResponse tokens: {response_tokens}")
    return response['choices'][0]['message']['content'][:100]


# Infinite loop to call ChatGPT and calculate cost
while True:
    try:
        # Use ThreadPoolExecutor to make 3 simultaneous calls
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(make_call) for _ in range(5)]
            results = [future.result() for future in futures]

        # Print the responses
        for result in results:
            print(f"Response: {result}...")

        print(f"Total cost so far: ${total_cost:.4f}")
        if total_cost > 7:
            break

        # Delay to stay within rate limits
        time.sleep(2.73)

    except openai.error.OpenAIError as e:
        print(f"Error: {str(e)}")
