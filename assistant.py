from yandex_cloud_ml_sdk import YCloudML
from search_system import upload_data, make_search_tool


class Assistant:
    def __init__(self, folder_id: str, api_key: str):
        self.folder_id = folder_id
        self.api_key = api_key
        self.init_sdk()
        self.init_model()
        uploaded_files = upload_data(self.sdk, "data")
        self.search_tool = make_search_tool(self.sdk, uploaded_files)
        self.init_assistant(
            instruction="Ты - работник приёмной комиссии Московского Авиационного Института. "
            "Система автоматически напомнит тебе релевантную информацию по вопросу, тебе необходимо "
            "внимательно с ней ознакомиться и ответить на вопрос. Если в предоставленной информации нет "
            "ответа на вопрос, честно скажи пользователю, что информации не найдено, или задай ему уточняющий вопрос.",
            tools=[self.search_tool]
        )

    def init_sdk(self):
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        return self.sdk

    def init_model(self):
        self.model = self.sdk.models.completions("yandexgpt", model_version="rc")
        return self.model

    def init_assistant(self, instruction, tools = None):
        kwargs = {}
        if tools and len(tools) > 0:
            kwargs = {"tools": tools}
        self.assistant = self.sdk.assistants.create(
            self.model, ttl_days=1, expiration_policy="since_last_active", **kwargs
        ).update(instruction=instruction)
        return self.assistant

    def init_thread(self):
        self.thread = self.sdk.threads.create(ttl_days=1, expiration_policy="static")
        return self.thread

    def start_chat(self):
        self.thread = self.sdk.threads.create(ttl_days=1, expiration_policy="static")
    
    def end_chat(self):
        self.thread.delete()

    def ask(self, message: str):
        self.thread.write(message)
        run = self.assistant.run(self.thread)
        return run.wait().text
