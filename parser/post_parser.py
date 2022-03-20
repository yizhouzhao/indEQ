import json

class VideoCausalityParser():
    def __init__(self) -> None:
        pass

    def parse_video_answer(self, answer_dict):
        """
        Parse video answer
        """

        description = ""
        qas = answer_dict["question_answer"]

        # description += f"{qas[0][2][0]} people in the {qas[1][2][0]}. "
        description += f"the environment feels {qas[2][2][0]} and {qas[2][2][1]}. "
        
        how_many_people = int(qas[0][2][0]) if qas[0][2][0].isdigit() else 0

        if how_many_people == 0:
            description += f"nobody is in the {qas[1][2][0]}. " 
        elif how_many_people == 1:
            description += f"one person is in the {qas[1][2][0]}. "
            description += f"the person looks like {qas[3][2][0]} or {qas[1][2][1]} and feels like {qas[4][2][0]} and {qas[4][2][1]}. "
            description += f"the person is {qas[5][2][0]} or {qas[5][2][1]}. "
        elif how_many_people == 2:
            description += f"two people are in the {qas[1][2][0]}. "
            description += f"the person on the left looks like {qas[6][2][0]} or {qas[6][2][1]} and feels like {qas[7][2][0]} and {qas[7][2][1]}. "
            description += f"the person on the right looks like {qas[8][2][0]} or {qas[8][2][1]} and feels like {qas[9][2][0]} and {qas[9][2][1]}. "
            description += f"they are {qas[10][2][0]} or {qas[10][2][1]}. "
        else:
            description += f"a group of people are in the {qas[1][2][0]}. "
            description += f"they are {qas[10][2][0]} or {qas[10][2][1]}. "
        
        answer_dict["description"] = description
        return
            
class TextRuleParser():