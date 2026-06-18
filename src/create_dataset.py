"""
Script for creating various datasets from the CNN/DailyMail dataset.

Setting CREATE_DATASET to True will create a subset of the CNN/DailyMail dataset.
Setting CONVERT_DATASET to True will perform simple prompt adaptations to the dataset 
instances (by uncommenting the desired adaptation code).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, load_from_disk
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM



model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
untied_heads = False
exit_layers=[7, 15, 23, 31]
ee_softmax_threshold=[0.7, 0.6, 0.5, 0.4]       # lower thresholds without sampling and cache

tokenizer = AutoTokenizer.from_pretrained(model_id)

CREATE_DATASET = False
if CREATE_DATASET:
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    # Reduce to only short instances of the dataset
    print("Filtering dataset")
    dataset = dataset.filter(lambda row: len(tokenizer.encode(row['article']+row['highlights']))<400)
    print("Whole dataset length:", len(dataset))

    keep_only = 32
    dataset = dataset.shuffle(seed=42).select(range(keep_only, keep_only+20))

    # dataset.save_to_disk(f"../datasets/cnn-dm_validation_short-shuffled{keep_only}")
    dataset.to_json(f"../datasets/cnn-dm_validation_short-shuffled{keep_only}-{keep_only+20}.json")


def convert_dataset(data_file, data_dir, model_id, out_file_suffix, system_prompt=None, convert_fct=None, test=False):
    dataset = load_dataset("json", data_dir=data_dir, data_files={"validation": data_file}, split="validation")

    def update_field(dataset, id, column, value):
        def update(instance):
            if instance["id"]==id:
                instance[column]=value
            return instance
        return dataset.map(update)

    if system_prompt != None:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=0,
        )

    if test:
        dataset = dataset.select(range(0, 2))

    if system_prompt != None:
        dataset = dataset.map( \
            lambda row: { \
            "article": row["article"], "highlights": row["highlights"], "id": row["id"], \
            "prompt_messages": [ \
                {"role": "system", "content": system_prompt}, \
                {"role": "user", "content": row["article"]}, \
            ]})

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for i, outputs in enumerate(tqdm(pipeline( \
                KeyDataset(dataset, "prompt_messages"), \
                max_new_tokens=512, \
                eos_token_id=terminators, \
                do_sample=False, \
                temperature=0.6, \
                top_p=0.9, \
            ), total=len(dataset))):

            dataset = update_field(dataset, dataset[i]["id"], "article", outputs[0]["generated_text"][-1]['content'])
        
        dataset = dataset.remove_columns(["prompt_messages"])
        
    if convert_fct != None:
        for i in tqdm(range(len(dataset))):
            article = dataset[i]["article"]

            article = convert_fct(article)
            dataset = update_field(dataset, dataset[i]["id"], "article", article)


    if test:
        print(dataset[0])
        print(dataset[1])
    else:
        data_file_name = data_file.replace(".json", f"{out_file_suffix}.json")
        dataset.to_json(f"{data_dir}{data_file_name}")

CONVERT_DATASET = False
if CONVERT_DATASET:
    data_file = "cnn-dm_validation_short-shuffled32-52.json"
    data_dir = "../datasets/"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ## Create copy
    # def convert_fct(article):
    #     return article
    # convert_dataset(data_file, data_dir, model_id, "_sentences_exclamation_mark_tagged", convert_fct=convert_fct, test=False)

    ### Remove unnecessary words
    # system_prompt = "The user will give you a news article and you remove unnecessary words like stop words. Please remove all words that are not necessary for the content and copy all other words as they are with at most slight grammatical modifications. Don't add any new words. Don't change the order of words."
    # convert_dataset(data_file, data_dir, model_id, system_prompt=system_prompt)

    ### Remove stop_words
    # import nltk
    # from nltk.corpus import stopwords
    # from nltk.tokenize import word_tokenize

    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('punkt_tab')

    # # Get English stopwords and tokenize
    # stop_words = set(stopwords.words('english'))

    # def convert_fct(article):
    #     tokens = word_tokenize(article.lower())

    #     # Remove stopwords
    #     filtered_tokens = [word for word in tokens if word not in stop_words]

    #     article = " ".join(filtered_tokens)
    #     return article

    # convert_dataset(data_file, data_dir, model_id, convert_fct=convert_fct)

    ### Use frequent synonyms
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace each word by a more common/frequent synonym if there is any. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_frequent_synonyms", system_prompt=system_prompt, test=False)


    # ### Use simple synonyms
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace each word by a simpler synonym (i.e. more frequent or suitable for simple language) if there is any. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_simple_synonyms", system_prompt=system_prompt, test=False)

    ### Randomly replace letters
    # replace_fraction = 0.05
    # import random
    # import string
    # def convert_fct(article):
    #     weights = torch.ones((len(article),))

    #     non_alpha_indices = [i for i, x in enumerate(article) if not x.isalpha()]
    #     weights[non_alpha_indices] = 0
    #     replace_indices = torch.multinomial(weights, int(replace_fraction*len(article)))
    #     for i in replace_indices:
    #         article = article[:i] + random.choice(string.ascii_letters) + article[i+1:]

    #     return article

    # convert_dataset(data_file, data_dir, model_id, "_letters_replaced", convert_fct=convert_fct, test=False)


    # ### Use familiar language
    # system_prompt = "The user will give you a news article and you should replace certain words. Please transform the text into more familiar instead of formal language. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_familiar language", system_prompt=system_prompt, test=False)


    # ### Use formal language
    # system_prompt = "The user will give you a news article and you should replace certain words. Please transform the text into more formal and distinct instead of familiar language. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_formal_language", system_prompt=system_prompt, test=False)

    # ### Use distinctive synonyms
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace each word by a more distinctive/particular synonym if there is any. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_distinctive_synonyms", system_prompt=system_prompt, test=False)

    # ### Insert emojis 
    # system_prompt = "The user will give you a news article and you should enhance the text by appending a lot of emojis to its words. Please for each word that expresses some sentiment, add a suitable emoji after the word. You may exaggarate a bit regarding their expressiveness. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_emojis_inserted", system_prompt=system_prompt, test=False)

#     ### Insert sentimental smileys
#     smiley_list = """☺️🙂😊😀😁	Happiness
# 😃😄😎	Fun, cool
# 😆😂 	Laughing, fun
# ☹️🙁😞😟😣😖	Frown, sad, pouting 
# 😢😭	Crying, sadness 
# 🥲🥹😂	Tears of happiness
# 😠😡 	Angry
# 😨😧😦😱😫😩	Horror, disgust, sadness, great dismay 
# 😮😯😲 	Surprise, shock 
# 😗😙😚😘😍	Kiss, love, gratitude
# 🫤🤔😕😟	Skeptical, annoyed, undecided, uneasy, hesitant
# 😳😞😖	Embarrassed, blushing
# 😇	Angel, halo, saint, innocent 
# 😈	Evil, devilish
# 😵😵‍💫😕🤕	Drunk, confused 
# 🤒😷🤢	Being sick
# 🤨 	Scepticism, disbelief, disapproval
# 😬	Grimacing, nervous, awkward"""
#     system_prompt = "The user will give you a news article and you should enhance the text by appending a lot of smileys to its words. Please for each word consider its sentiment expressed and add a suitable smiley behind the word if it expresses some kind of sentiment. For example, change \"interested\" to \"interested 😀\", \"grenade\" to \"grenade ☹️\", \"birthday\" to \"birthday 🥳\", \"sigh\" to \"sigh 😢\", \"angry\" to \"angry 🤬\". Prefer simple and most common smilieys. You may exaggarate a bit regarding their expressiveness. Here is an example. Article: (CNN) Travelers and critical cargo in Egypt could soon be moving faster thanks to high-speed trains recently unveiled by Siemens Mobility. Your answer: Travelers 😀 and critical 🤨 cargo in Egypt could soon be moving faster 😮 thanks 😚 to high-speed trains recently unveiled 😮 by Siemens Mobility. Here is a list of smileys with the corresponding sentiment they express, which you can use: " + smiley_list + "\n\nReply with nothing but the modified article text."
#     convert_dataset(data_file, data_dir, model_id, "_smileys_inserted", system_prompt=system_prompt, test=False)


    # ### All uppercase
    # def convert_fct(article):
    #     return article.upper()

    # convert_dataset(data_file, data_dir, model_id, "_uppercase", convert_fct=convert_fct, test=False)


    # ### nouns uppercase
    # import nltk
    # nltk.download('averaged_perceptron_tagger_eng')
    # def convert_fct(article):
    #     is_noun = lambda pos: pos[:2] == 'NN'
    #     tokenized = nltk.word_tokenize(article)
    #     tokens = [word.upper() if is_noun(pos) else word for (word, pos) in nltk.pos_tag(tokenized)] 
    #     return " ".join(tokens)

    # convert_dataset(data_file, data_dir, model_id, "_nouns_uppercase", convert_fct=convert_fct, test=False)


    # ### !-tagged nouns / verbs
    # import nltk
    # nltk.download('averaged_perceptron_tagger_eng')
    # def convert_fct(article):
    #     # is_noun = lambda pos: pos[:2] == 'NN'
    #     is_pos = lambda pos: pos[:2] == 'VB'
    #     tokenized = nltk.word_tokenize(article)
    #     tokens = [word + " (!)" if is_pos(pos) else word for (word, pos) in nltk.pos_tag(tokenized)] 
    #     return " ".join(tokens)

    # # convert_dataset(data_file, data_dir, model_id, "_nouns_exclamation_mark_tagged", convert_fct=convert_fct, test=False)
    # convert_dataset(data_file, data_dir, model_id, "_verb_exclamation_mark_tagged", convert_fct=convert_fct, test=False)


    # ### Make it extreme
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace words such that they are more extreme in what they are saying. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_extreme", system_prompt=system_prompt, test=False)

    # ### Use extreme words
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace words to more extreme representatives. Preserve the meaning. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_extreme_words", system_prompt=system_prompt, test=False)

    # ### !-tagged nouns / verbs in important sentences
    # data_file = "cnn-dm_validation_short-shuffled32-52_sentences_exclamation_mark_tagged.json"
    # import nltk
    # import re
    # nltk.download('averaged_perceptron_tagger_eng')
    # pattern = re.compile('\(!\)(?=([^!]|\(!\))*\.)')  # Find all (!) of which the sentence ends with a period (not with !)
    # def convert_fct(article):
    #     # At first, tag all nouns/verbs, then remove all the (!) of nouns/verbs 
    #     # that are not in important sentences.
    #     is_pos = lambda pos: pos[:2] == 'NN'
    #     # is_pos = lambda pos: pos[:2] == 'VB'
    #     tokenized = nltk.word_tokenize(article)
    #     tokens = [word + " (!)" if is_pos(pos) else word for (word, pos) in nltk.pos_tag(tokenized)] 
    #     article = " ".join(tokens)
    #     article = pattern.sub('', article) # delete all unwanted (!)
    #     return article

    # convert_dataset(data_file, data_dir, model_id, "_nouns_in_important_sentences_tagged", convert_fct=convert_fct, test=False)

    # ### Combination: First remove unnecessary words, then use frequent but distinctive synonyms.
    # # system_prompt = "The user will give you a news article and you should stringently replace and remove certain words. Please remove all words that are not necessary for the content. At the same time, please replace each remaining word by a more common/frequent but also distinctive/particular synonym if there is any. Copy all other words as they are. Try to drop at least 15 % of words but don't add any new words and don't change the order of words! Reply with nothing but the modified article text."
    # system_prompt = "The user will give you a news article and you should stringently remove about 15 % of words or more. Please drop all words that are not necessary to understand the content like adjectives and stop words and copy all other words as they are with at most slight grammatical modifications. Don't add any new words. Don't change the order of words. Reply with nothing but the modified article text."
    # convert_dataset(data_file, data_dir, model_id, "_removed_unnecessary_words_II", system_prompt=system_prompt, test=False)
    # data_file = "cnn-dm_validation_short-shuffled32-52_removed_unnecessary_words_II.json"
    # system_prompt = "The user will give you a news article and you should replace certain words. Please replace each word by a more common/frequent but also distinctive/particular synonym if there is any. Reply with nothing but the modified article text."
    # # convert_dataset(data_file, data_dir, model_id, "_removed_unnecessary_words_frequent_distinctive_synonyms", system_prompt=system_prompt, test=False)
    # convert_dataset(data_file, data_dir, model_id, "_frequent_distinctive_synonyms", system_prompt=system_prompt, test=False)


    ### Sentences concatenated by and
    data_file = "cnn-dm_validation_short-shuffled32-52_removed_unnecessary_words_II_frequent_distinctive_synonyms.json"
    import nltk
    import re
    pattern = re.compile('\.("?) ("?)(?=[A-Z])')  # Find all ". <Capital-letter>" (possibly with a quote sign before or after the dot) to replace the ". " by " and "
    def convert_fct(article):
        article = pattern.sub("\g<1> and \g<2>", article) 
        return article
    convert_dataset(data_file, data_dir, model_id, "_sentences_concatenated_by_and", convert_fct=convert_fct, test=False)


CREATE_TOKENIZED_DATAEST = False
if CREATE_TOKENIZED_DATAEST:
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "facebook/layerskip-llama3.2-1B"

    # copy_chat_template_model_id = None
    copy_chat_template_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EeLlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=0,

    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if copy_chat_template_model_id is not None:
        # Add chat template to non-chat model
        tokenizer.eos_token_id = model.config.eos_token_id
        chat_template = AutoTokenizer.from_pretrained(copy_chat_template_model_id).chat_template
        chat_template = chat_template.replace("<|eot_id|>", tokenizer.decode(model.config.eos_token_id))  # Ensure that the chat template uses the correct eos token id
        tokenizer.chat_template = chat_template

    print(f"Device: {device} / {model.device}")

    
    dataset = load_from_disk("../datasets/cnn-dm_short")
    dataset = dataset.rename_column("article", "prompt")
    dataset = dataset.rename_column("highlights", "completion")
    dataset = dataset.shuffle(seed=42)
    # dataset = dataset.select(range(10205, 15307))
    dataset = dataset.select(range(9712, 19424))   # Second half of cnn-dm_short

    dataset0 = load_from_disk("../datasets/eellama-3p2-1B-layerskip/cnn-dm_short_full_model_generations_10205")
    dataset0 = dataset0.select(range(9712))   # First half of cnn-dm_short

    # Tokenize prompt and completion and save them as input_ids and output_ids.
    print("Generating dataset")

    generated_data = []
    for i in tqdm(range(len(dataset))):
        # First tokenize only user prompt, to get the length of the input part
        messages_prompt_only = [
            {"role": "user", "content": dataset[i]["prompt"]},
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            messages_prompt_only,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False
        ).to(model.device)        # shape (1, sequence_length)
        prompt_length = prompt_tokens.shape[1]

        # Now tokenizer prompt+completion and split into input and output ids based 
        # on the prompt length
        messages = [
            {"role": "user", "content": dataset[i]["prompt"]},
            {"role": "assistant", "content": dataset[i]["completion"]},
        ]
        tokenizer_result = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        token_ids = tokenizer_result["input_ids"].squeeze(0)  # squeezes from shape (1, sequence_length) to (sequence_length,)

        input_ids = token_ids[:prompt_length]
        output_ids = token_ids[prompt_length:]

        if i==0:
            assert torch.equal(input_ids, prompt_tokens.squeeze(0)), "Tokenizing prompt+completion does not equal tokenizing prompt and tokenizing completion concatenated."

        generated_data.append({
            "input_ids": input_ids, 
            "output_ids": output_ids
        })



    import pdb; pdb.set_trace()
    # Attention: The following lines required an update and are not tested:
    dataset_generated = Dataset.from_list(generated_data)
    dataset = concatenate_datasets([dataset0, dataset_generated])
    dataset.save_to_disk(f"../datasets/eellama-3p2-1B-layerskip/cnn-dm_short_mixed_full_model_generations_and_labels")


GATHER_STATISTICS = True
def dataset_statistics():

    data_dir = "../datasets/"
    data_file = "cnn-dm_validation_short-shuffled32-52.json"

    print("Loading dataset from disk")
    if data_dir[-1] != "/": data_dir += "/"

    if data_file.endswith(".json"):
        dataset = load_dataset("json", data_dir=data_dir, data_files={"validation": data_file}, split="validation")
    else:
        import pdb; pdb.set_trace()
        dataset = load_from_disk(f"{data_dir}{data_file}")

    print(f"Dataset length: {len(dataset)}")

    print(dataset[2])

    import pdb; pdb.set_trace()
if GATHER_STATISTICS:
    dataset_statistics()