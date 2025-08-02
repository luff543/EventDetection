from Prompt import Prompt
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptForClassification

class HardPrompt(Prompt):
    def __init__(self, model_name, checkpoint, inference=False):
        super().__init__(model_name, checkpoint, inference)
        self.template = ManualTemplate(
                        tokenizer=self.tokenizer,
                        text='标题：{"meta":"title"} 网页正文：{"meta":"message","shortenable":True} 问题：上述内容是不是活动网页？ 答案:{"mask"}',
                        )
        self.verbalizer = ManualVerbalizer(
                            classes=self.classes,
                            label_words={
                                "不是": ["不", "否"],
                                "是的": ["是"],
                            },
                            tokenizer=self.tokenizer,
                        )
        self.model = PromptForClassification(
                        template=self.template,
                        plm=self.plm,
                        verbalizer=self.verbalizer,
                        freeze_plm = False
                        )