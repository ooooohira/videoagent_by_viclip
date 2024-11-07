import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

from viclip_retrieve import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


client = OpenAI()


def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview"
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    cached_value = get_from_cache(key)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    # print("Not hit cache", key)
    input()

    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps and each segment contains consecutive 4 frames(4 seconds). Given the following information of the sampled segment in the video:
    {caption}
    The correct query and answer for each segment in the video, created by humans, is shown above.
    Please answer the following question considering above: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


# def generate_description_step(question, caption, num_frames, segment_des):
#     formatted_description = {
#         "frame_descriptions": [
#             {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
#             {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
#             {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
#         ]
#     }
#     prompt = f"""
#     Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
#     {caption}
#     #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
#     #O to denote that the sentence is an action done by someone other than the camera wearer.
#     To answer the following question: 
#     ``` 
#     {question}
#     ``` 
#     However, the information in the initial frames is not suffient.
#     Objective:
#     Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
#     To achieve this, we will:
#     1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
#     2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
#     For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
#     Select multiple frames from one segment if necessary to gather comprehensive insights. 
#     Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
#     ```
#     {formatted_description}
#     ```
#     """
#     system_prompt = "You are a helpful assistant designed to output JSON."
#     response = get_llm_response(system_prompt, prompt, json_format=True)
#     return response

# # クリア
# def generate_description_step(question, caption, num_frames, segment_des):
#     formatted_description = {
#         "frame_descriptions": [
#             {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx", "question": "xxx?", "pred_answer": "xxx"},
#             {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx", "question": "xxx?", "pred_answer": "xxx"},
#             {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx", "question": "xxx?", "pred_answer": "xxx"},
#         ]
#     }
#     prompt = f"""
#     Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following QA-formatted descriptions of the sampled segment which is composed of 4 frames, centered around that frame, with the preceding 3 frames and the following 4 frames in the video:
#     {caption}
#     The correct QA for each segment in the video, created by humans, is shown above.
#     To answer the following question: 
#     ``` 
#     {question}
#     ``` 
#     However, the information in the initial segments is not suffient.
#     Objective:
#     Our goal is to identify additional segments that contain crucial information necessary for answering the question. These segments should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
#     To achieve this, we will:
#     1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
#     2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
#     3. Generate questions and predicted answers to extract the necessary information from those segments.
#     For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
#     Select multiple frames from one segment if necessary to gather comprehensive insights. 
#     Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
#     ```
#     {formatted_description}
#     ```
#     """
#     system_prompt = "You are a helpful assistant designed to output JSON."
#     response = get_llm_response(system_prompt, prompt, json_format=True)
#     return response
# クリア
def generate_description_step(question, caption, num_frames, segment_des):
    formatted_description = {
        "frame_descriptions": [
            {"start frame number": "xxx", "end frame number": "xxx", "description": "video of xxx", "query": "xxx?"},
            {"start frame number": "xxx", "end frame number": "xxx", "description": "video of xxx", "query": "xxx?"},
            {"start frame number": "xxx", "end frame number": "xxx", "description": "video of xxx", "query": "xxx?"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps and each segment contains consecutive 4 frames(4 seconds). Given the following information of the sampled segment in the video:
    {caption}
    The correct query and answer for each segment in the video, created by humans, is shown above.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial segments is not suffient.
    Objective:
    Our goal is to identify additional segments that contain crucial information necessary for answering the question. These segments should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Determine the range that includes 8 or more frames of frames that contain segments that are most relevant to the question. These segments should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    2. Generate queries which is input to vision and language model to extract the necessary information from those segments. The query ranges from requesting captions for entire segments to asking detailed questions about specific content. Output the necessary information.
    For each segment identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per segment. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions, the start frame number and end frame number which express range and query in JSON format, note the range includes 8 or more frames. if start frame number is 1 and end frame number is 7, then the range contains only 7 frames, so it is forbidden. The range should include 8 or more frames.
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response
#クリア
def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response

#クリア
def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps and each segment contains consecutive 4 frames(4 seconds). Given the following information of the sampled segment in the video:
    {caption}
    The correct query and answer for each segment in the video, created by humans, is shown above.
    Please answer the following question considering above: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response

#クリア
def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps and each segment contains consecutive 4 frames(4 seconds). Given the following information of the sampled segment in the video:
    {caption}
    The correct query and answer for each segment in the video, created by humans, is shown above.
    Please answer the following question considering above: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response



# def read_caption(captions, sample_idx):
#     video_caption = {}
#     for idx in sample_idx:
#         video_caption[f"frame {idx}"] = captions[idx - 1]
#     return video_caption
from viclip_retrieve import search_best_segment
from video_frame import split_video_to_segments
from video_qa import qa_generation
# 0 index を想定
def extract_frames_index(full_frames, sample_idx):
    start_idx = max(0, sample_idx - 3)
    left_padding = 3 - sample_idx if sample_idx < 4 else 0
    end_idx = min(len(full_frames) - 1, sample_idx + 4)
    right_padding = 4 - (len(full_frames) - 1 - sample_idx) if sample_idx + 4 >= len(full_frames) else 0
    frames = (
        [full_frames[0]] * left_padding +
        full_frames[start_idx:sample_idx + 1] +
        full_frames[sample_idx + 1:end_idx + 1] +
        [full_frames[-1]] * right_padding
    )
    return frames
# 1 index を想定    
def get_ans_frames(prompt, sub_frames, sample_idx, qa_result):
    ans = qa_generation(sub_frames, prompt)
    key = f"frame {sample_idx-3+1}~{sample_idx+4}"
    if key in qa_result:
        qa_result[key][prompt] = ans
    else:
        qa_result[key] = {}
        qa_result[key][prompt] = ans
    return qa_result
# 1 index を想定    
def get_ans_index(prompt, sample_idx, full_frames, qa_result):
    sub_frames = extract_frames_index(full_frames, sample_idx-1)
    ans = qa_generation(sub_frames, prompt)
    key = f"frame {sample_idx-3+1}~{sample_idx+4}"
    if key in qa_result:
        qa_result[key][prompt] = ans     
    else:
        qa_result[key] = {}
        qa_result[key][prompt] = ans    
    return qa_result    


# frame_index_listは最初は一様サンプリング、最初以外はframe_retrieval_seg_egoの結果(1-index)を想定
def memory_update(prompt_list, frame_index_list, full_frames, qa_result):
    for prompt, sample_idx in zip(prompt_list, frame_index_list):
        get_ans_index(prompt, sample_idx, full_frames, qa_result)
    qa_result = dict(sorted(qa_result.items(), key=lambda x: int(x[0].split(" ")[1].split("~")[0])))
    return qa_result


import random
global now
global hit
now = 0
hit = 0
def run_one_question(video_id, ann, logs):
    global now
    global hit
    timestamps, frames = split_video_to_segments("/experiment/data/videos/videos/" + video_id + ".mp4", 1)
    num_frames = len(frames)/10
    sample_idx = np.linspace(3, num_frames-1-4, num=5, dtype=int).tolist()
    # try:
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    qa_result = {} # memoryでsampled_capsに対応
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    # timestamps, frames = split_video_to_segments("/experiment/data/videos/videos/" + video_id + ".mp4", 1)


    # num_frames = len(frames)
    ### Step 1 ###
    
    # sample_idx = np.linspace(3, num_frames-1-4, num=5, dtype=int).tolist()
    prompt_list = ["Describe this segment." for i in range(5)]
    qa_result = memory_update(prompt_list, sample_idx, frames, qa_result)
    # print(qa_result)
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, qa_result, num_frames
    )
    answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str)

    ### Step 2 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                qa_result,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            # print(parsed_candiate_descriptions)
            frame_idx, q_list = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            # sampled_caps = read_caption(caps, sample_idx)
            qa_result = memory_update(q_list, frame_idx, frames, qa_result)
            # print(qa_result)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, qa_result, num_frames
            )
            answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, qa_result, num_frames
            )
            answer = parse_text_find_number(answer_str)

    ### Step 3 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                qa_result,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx, q_list = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 3: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            # sampled_caps = read_caption(caps, sample_idx)
            qa_result = memory_update(q_list, frame_idx, frames, qa_result)
            # print(qa_result)
            answer_str = generate_final_answer(
                formatted_question, qa_result, num_frames
            )
            answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"Step 3 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, qa_result, num_frames
            )
            answer = parse_text_find_number(answer_str)
    if answer == -1:
        logger.info("Answer Index Not Found!")
        answer = random.randint(0, 4)

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }
    if label == answer:
        hit += 1
    now += 1
    # print(qa_result)
    print(f"now : {now}, true : {label}, pred : {answer}, now acc : {hit/now}")
    # except:
    #     print("EEEEERRRROOOORRR")
    #     answer = random.randint(0, 4)
    #     label = int(ann["truth"])
    #     logs[video_id] = {
    #         "answer": answer,
    #         "label": int(ann["truth"]),
    #         "corr": int(label == answer),
    #         "count_frame": len(sample_idx),
    #     }        

from tqdm import tqdm
def main():
    # if running full set, change subset to fullset
    input_ann_file = "/experiment/data/subset_anno.json"
    # all_cap_file = "lavila_subset.json"
    json_file_name = "/experiment/data/egoschema_subset_not.json"
    anns = json.load(open(input_ann_file, "r"))
    # all_caps = json.load(open(all_cap_file, "r"))
    logs = {}
    # print(len(list(anns.keys())))
    # for video_id in tqdm(list(anns.keys())[3:]):
    #     run_one_question(video_id, anns[video_id], logs)
    #     print(logs)
    tasks = [
        (video_id, anns[video_id], logs)
        for video_id in list(anns.keys())
    ][100:200]
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     executor.map(lambda p: run_one_question(*p), tasks)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks)))
    json.dump(logs, open(json_file_name, "w"))



if __name__ == "__main__":
    print("latest no print")
    main()
