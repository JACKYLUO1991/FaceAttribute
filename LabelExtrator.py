import json
import os


class FaceLabel:
    """
    106 x,y label
    [(x1,y1), (x2,y2)...]
    """
    # __landmark = []
    # __landmark_schema = []

    """left,height,top,width"""
    # __face_rectangle = []
    # __face_rectangle_schema = []

    """
    BLACK, WHITE, ASIAN
    """
    # __ethnicity = []
    # __ethnicity_schema = []

    """
    "close": 100.0,
    "other_occlusion": 0.0,
    "open": 0.0,
    "surgical_mask_or_respirator": 0.0
    surgical_mask_or_respirator：嘴部被医用口罩或呼吸面罩遮挡的置信度
    other_occlusion：嘴部被其他物体遮挡的置信度
    close：嘴部没有遮挡且闭上的置信度
    open：嘴部没有遮挡且张开的置信度
    """
    # __mouth_status = []
    # __mouth_status_schema = []

    """
    "eyestatus": {
     "left_eye_status": {
      "occlusion": 0.0,
      "normal_glass_eye_close": 0.0,
      "dark_glasses": 0.0,
      "no_glass_eye_close": 0.0,
      "no_glass_eye_open": 99.999,
      "normal_glass_eye_open": 0.0
     },
     "right_eye_status": {
      "occlusion": 0.02,
      "normal_glass_eye_close": 0.0,
      "dark_glasses": 0.002,
      "no_glass_eye_close": 0.0,
      "no_glass_eye_open": 99.964,
      "normal_glass_eye_open": 0.014
     }
    嘴部状态信息，包括以下字段。每个字段的值都是一个浮点数，范围 [0,100]，小数点后 3 位有效数字。字段值的总和等于 100。
    no_glass_eye_open：不戴眼镜且睁眼的置信度
    normal_glass_eye_close：佩戴普通眼镜且闭眼的置信度
    normal_glass_eye_open：佩戴普通眼镜且睁眼的置信度
    dark_glasses：佩戴墨镜的置信度
    no_glass_eye_close：不戴眼镜且闭眼的置信度
    """
    # __left_eye_status = []
    # __left_eye_status_schema = []
    # __right_eye_status = []
    # __right_eye_status_schema = []

    """
    None    不佩戴眼镜
    Dark    佩戴墨镜
    Normal  佩戴普通眼镜
    """
    # __glass = []
    # __glass_schema = []

    """
    health：健康
    stain：色斑
    acne：青春痘
    dark_circle：黑眼圈
    """
    # __skinstatus = []
    # __skinstatus_schema = []

    """
    Male    男性
    Female  女性
    """
    # __gender=[]
    # __gender_schema = []

    """
    value：值为人脸的质量判断的分数，是一个浮点数，范围 [0,100]，小数点后 3 位有效数字。
    threshold：表示人脸质量基本合格的一个阈值，超过该阈值的人脸适合用于人脸比对。
    """
    # __facequality = 0.0

    """
    "happiness": 0.017,
    "sadness": 0.0,
    "anger": 0.001,
    "neutral": 99.978,
    "surprise": 0.004,
    "fear": 0.0,
    "disgust": 0.0
    情绪识别结果。返回值包含以下字段。每个字段的值都是一个浮点数，范围 [0,100]，小数点后 3 位有效数字。每个字段的返回值越大，则该字段代表的状态的置信度越高。字段值的总和等于 100。
    anger：愤怒
    disgust：厌恶
    fear：恐惧
    happiness：高兴
    neutral：平静
    sadness：伤心
    surprise：惊讶
    """
    # __emotion = []
    # __emotion_schema = []

    """ 
    年龄分析结果。返回值为一个非负整数。
    """
    # __age = 0

    """
    人脸模糊分析结果。返回值包含以下属性：
    motionblur：人脸移动模糊度分析结果。
    gaussianblur：人脸高斯模糊度分析结果。
    blurness：新的人脸模糊分析结果。
    每个属性都包含以下字段：
    value 的值为是一个浮点数，范围 [0,100]，小数点后 3 位有效数字。
    threshold 表示人脸模糊度是否影响辨识的阈值。
    "blur": {
     "blurness": {
      "value": 0.109,
      "threshold": 50.0
     },
     "gaussianblur": {
      "value": 0.109,
      "threshold": 50.0
     },
     "motionblur": {
      "value": 0.109,
      "threshold": 50.0
     }
    },
    """
    # __blurness = 0.0
    # __gaussianblur = 0.0
    # __motionblur = 0.0

    """
    "headpose": {
     "yaw_angle": 13.14796,
     "pitch_angle": -4.469762,
     "roll_angle": -4.9755783
    },
    pitch_angle：抬头
    roll_angle：旋转（平面旋转）
    yaw_angle：摇头
    """
    # __headpose = []
    # __headpose_schema = []

    """
    笑容分析结果。返回值包含以下属性：
    value：值为一个 [0,100] 的浮点数，小数点后3位有效数字。数值越大表示笑程度高。
    threshold：代表笑容的阈值，超过该阈值认为有笑容。
    """
    # __smile = 0.0

    """
    "eyegaze": {
     "right_eye_gaze": {
      "position_y_coordinate": 0.428,
      "vector_z_component": 0.993,
      "vector_x_component": 0.039,
      "position_x_coordinate": 0.496,
      "vector_y_component": -0.109
     },
     "left_eye_gaze": {
      "position_y_coordinate": 0.432,
      "vector_z_component": 0.989,
      "vector_x_component": 0.022,
      "position_x_coordinate": 0.489,
      "vector_y_component": -0.147
     }
    },
    眼球位置与视线方向信息。返回值包括以下属性：
    left_eye_gaze：左眼的位置与视线状态
    right_eye_gaze：右眼的位置与视线状态
    每个属性都包括以下字段，每个字段的值都是一个浮点数，小数点后 3 位有效数字。
    position_x_coordinate: 眼球中心位置的 X 轴坐标。
    position_y_coordinate: 眼球中心位置的 Y 轴坐标。
    vector_x_component: 眼球视线方向向量的 X 轴分量。
    vector_y_component: 眼球视线方向向量的 Y 轴分量。
    vector_z_component: 眼球视线方向向量的 Z 轴分量。
    """
    # __right_eye_gaze = []
    # __right_eye_gaze_schema = []
    # __left_eye_gaze = []
    # __left_eye_gaze_schema = []

    """
    颜值识别结果。返回值包含以下两个字段。每个字段的值是一个浮点数，范围 [0,100]，小数点后 3 位有效数字。
    male_score：男性认为的此人脸颜值分数。值越大，颜值越高。
    female_score：女性认为的此人脸颜值分数。值越大，颜值越高。
    """
    # __male_score = 0.0
    # __female_score = 0.0

    def __init__(self, faceDict):
        # For multiple faces, face++ doesn't return all landmarks due to upper limitation, so we need to handle this exception
        if 'landmark' not in faceDict:
            self.__hasFace = False
            return
        else:
            self.__hasFace = True
        self.__landmark = [(value['x'], value['y']) for value in [
            faceDict['landmark'][index] for index in sorted(faceDict['landmark'].keys())]]
        self.__landmark_schema = self.__getSchema(faceDict['landmark'])

        self.__face_rectangle = self.__getValue(faceDict['face_rectangle'])
        self.__face_rectangle_schema = self.__getSchema(
            faceDict['face_rectangle'])

        attributes = faceDict['attributes']
        transResult = self.__ethnicityTrans(attributes['ethnicity']['value'])
        self.__ethnicity = transResult[0]
        self.__ethnicity_schema = transResult[1]
        self.__mouth_status = self.__getValue(attributes['mouthstatus'])
        self.__mouth_status_schema = self.__getSchema(
            attributes['mouthstatus'])
        self.__left_eye_status = self.__getValue(
            attributes['eyestatus']['left_eye_status'])
        self.__left_eye_status_schema = self.__getSchema(
            attributes['eyestatus']['left_eye_status'])
        self.__right_eye_status = self.__getValue(
            attributes['eyestatus']['right_eye_status'])
        self.__right_eye_status_schema = self.__getSchema(
            attributes['eyestatus']['right_eye_status'])

        transResult = self.__glassTrans(attributes['glass']['value'])
        self.__glass = transResult[0]
        self.__glass_schema = transResult[1]

        self.__facequality = attributes['facequality']['value']

        self.__skinstatus = self.__getValue(attributes['skinstatus'])
        self.__skinstatus_schema = self.__getSchema(attributes['skinstatus'])

        transResult = self.__genderTrans(attributes['gender']['value'])
        self.__gender = transResult[0]
        self.__gender_schema = transResult[1]

        self.__emotion = self.__getValue(attributes['emotion'])
        self.__emotion_schema = self.__getSchema(attributes['emotion'])

        self.__age = attributes['age']['value']
        self.__blurness = attributes['blur']['blurness']['value']
        self.__gaussianblur = attributes['blur']['gaussianblur']['value']
        self.__motionblur = attributes['blur']['motionblur']['value']

        self.__headpose = self.__getValue(attributes['headpose'])
        self.__headpose_schema = self.__getSchema(attributes['headpose'])
        self.__smile = attributes['smile']['value']

        self.__right_eye_gaze = self.__getValue(
            attributes['eyegaze']['right_eye_gaze'])
        self.__right_eye_gaze_schema = self.__getSchema(
            attributes['eyegaze']['right_eye_gaze'])
        self.__left_eye_gaze = self.__getValue(
            attributes['eyegaze']['left_eye_gaze'])
        self.__left_eye_gaze_schema = self.__getSchema(
            attributes['eyegaze']['left_eye_gaze'])
        self.__male_score = attributes['beauty']['male_score']
        self.__female_score = attributes['beauty']['female_score']

    # Private Method
    def __getSchema(self, dict):
        return [key for key in sorted(dict.keys())]

    def __getValue(self, dict):
        return [value for value in [dict[index] for index in sorted(dict.keys())]]

    def __ethnicityTrans(self, value):
        #BLACK, WHITE, ASIAN, INDIA
        schema = ['BLACK', 'WHITE', 'ASIAN', 'INDIA']
        if value == 'BLACK':
            result = [1, 0, 0, 0]
        elif value == 'WHITE':
            result = [0, 1, 0, 0]
        elif value == 'ASIAN':
            result = [0, 0, 1, 0]
        elif value == 'INDIA':
            result = [0, 0, 0, 1]
        else:
            print(value)
            raise TypeError
        return (result, schema)

    def __glassTrans(self, value):
        """
        None    不佩戴眼镜
        Dark    佩戴墨镜
        Normal  佩戴普通眼镜
        """
        schema = ['None', 'Dark', 'Normal']
        if value == 'None':
            result = [1, 0, 0]
        elif value == 'Dark':
            result = [0, 1, 0]
        elif value == 'Normal':
            result = [0, 0, 1]
        else:
            raise TypeError
        return (result, schema)

    def __genderTrans(self, value):
        """
        Male    男性
        Female  女性
        """
        schema = ['Male', 'Female']
        if value == 'Male':
            result = [1, 0]
        elif value == 'Female':
            result = [0, 1]
        else:
            raise TypeError
        return (result, schema)

    # Property method
    @property
    def Landmark(self):
        return self.__landmark

    @property
    def LandmarkSchema(self):
        return self.__landmark_schema

    @property
    def FaceRect(self):
        return self.__face_rectangle

    @property
    def FaceLandmarkRect(self):
        x = [x[0] for x in self.__landmark]
        y = [x[1] for x in self.__landmark]
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]

    @property
    def RectSchema(self):
        return self.__face_rectangle_schema

    @property
    def HasFace(self):
        return self.__hasFace

    @property
    def Ethnicity(self):
        return self.__ethnicity

    @property
    def EthnicitySchema(self):
        return self.__ethnicity_schema

    @property
    def MouthStatus(self):
        return self.__mouth_status

    @property
    def MouthStatusSchema(self):
        return self.__mouth_status_schema

    @property
    def LeftEyeStatus(self):
        return self.__left_eye_status

    @property
    def LeftEyeStatusSchema(self):
        return self.__left_eye_status_schema

    @property
    def RightEyeStatus(self):
        return self.__right_eye_status

    @property
    def RightEyeStatusSchema(self):
        return self.__right_eye_status_schema

    @property
    def Glass(self):
        return self.__glass

    @property
    def GlassSchema(self):
        return self.__glass_schema

    @property
    def FaceQuality(self):
        return self.__facequality

    @property
    def SkinStatus(self):
        return self.__skinstatus

    @property
    def SkinStatusSchema(self):
        return self.__skinstatus_schema

    @property
    def Gender(self):
        return self.__gender

    @property
    def GenderSchema(self):
        return self.__gender_schema

    @property
    def Emotion(self):
        return self.__emotion

    @property
    def EmotionSchema(self):
        return self.__emotion_schema

    @property
    def Age(self):
        return self.__age

    @property
    def Blurness(self):
        return self.__blurness

    @property
    def GaussianBlur(self):
        return self.__gaussianblur

    @property
    def MotionBlur(self):
        return self.__motionblur

    @property
    def HeadPose(self):
        return self.__headpose

    @property
    def HeadPoseSchema(self):
        return self.__headpose_schema

    @property
    def SmileValue(self):
        return self.__smile

    @property
    def SmileVector(self):
        if self.__smile > 50.0:
            return [1, 0]
        else:
            return [0, 1]

    @property
    def RightEyeGaze(self):
        return self.__right_eye_gaze

    @property
    def RightEyeGazeSchema(self):
        return self.__right_eye_gaze_schema

    @property
    def LeftEyeGaze(self):
        return self.__left_eye_gaze

    @property
    def LeftEyeGazeSchema(self):
        return self.__left_eye_gaze_schema

    @property
    def MaleScore(self):
        return self.__male_score

    @property
    def FemaleScore(self):
        return self.__female_score


class FaceLabels:
    def __init__(self, jsonPath):
        self.__faces = []
        if os.path.exists(jsonPath):
            result = json.load(open(jsonPath, "r"))
            if 'faces' in result:
                for face in result['faces']:
                    label = FaceLabel(face)
                    if label.HasFace:
                        self.__faces.append(label)

    @property
    def FaceNumber(self):
        return len(self.__faces)

    def getFace(self, index):
        return self.__faces[index]

    def getFaces(self):
        return self.__faces
