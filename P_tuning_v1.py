from Prompt import Prompt
from openprompt.prompts import PtuningTemplate,ManualVerbalizer,AutomaticVerbalizer
from openprompt import PromptForClassification

class P_tuning_v1(Prompt):
    def __init__(self, model_name, checkpoint, inference=False):
        super().__init__(model_name, checkpoint, inference)
        self.template = PtuningTemplate(
                            model=self.plm,
                            tokenizer=self.tokenizer,
                            text='{"soft":"网页标题:"}{"meta":"title","shortenable":20} {"soft":"网页正文:"}{"meta":"message","shortenable":True} {"soft":"问题:上述内容是不是活动网页？"} 答案:{"mask"}',
                        )
        self.verbalizer = AutomaticVerbalizer(
                            classes=self.classes,
                            tokenizer=self.tokenizer,
                        )        
                        # ManualVerbalizer(
                        #     classes=self.classes,
                        #     label_words=
                        #     {
                        #         "不是": ["不"],
                        #         "是的": ["是"],
                        #     },
                        #     tokenizer=self.tokenizer,
                        # )
        self.model = PromptForClassification(
                        template=self.template,
                        plm=self.plm,
                        verbalizer=self.verbalizer,
                        freeze_plm = False
                        )