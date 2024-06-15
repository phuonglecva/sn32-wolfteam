import json
import traceback
from abc import ABC


class AppConfig(ABC):

    def __init__(self, config_path='application.json'):
        self.config_path = config_path
        self.value = self.default_app_config()
        try:
            self.load_app_config()
        except Exception as e:
            print(e)
            traceback.print_exc()

    def default_app_config(self):
        print("start default_app_config")
        value = {
            "application": {
                "validator_pile": {
                    "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3": 2,
                    "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2": 25,
                    "5CaNj3BarTHotEK1n513aoTtFeXcjf6uvKzAyzNuv9cirUoW": 22,
                    "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": 10,
                    "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v": 7,
                    "5EhvL1FVkQPpMjZX4MAADcW42i3xPSF1KiCpuaxTYVr28sux": 18,
                    "5Dd8gaRNdhm1YP7G1hcB1N842ecAUQmbLjCRLqH5ycaTGrWv": 12,
                    "5DvTpiniW9s3APmHRYn8FroUWyfnLtrsid5Mtn5EwMXHN2ed": 14,
                    "5HbLYXUBy1snPR8nfioQ7GoA9x76EELzEq9j7F32vWUQHm1x": 24,
                    "5Hp18g9P8hLGKp9W3ZDr4bvJwba6b6bY3P2u3VdYf8yMR8FM": 9,
                    "5CXRfP2ekFhe62r7q3vppRajJmGhTi7vwvb2yr79jveZ282w": "",
                    "5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf": "50text",
                    "5CXC2quDN5nUTqHMkpP5YRp2atYYicvtUghAYLj15gaUFwe5": 5
                },
                "pile_server": {
                    "1": "http://localhost:8080/predict",
                    "2": "http://localhost:8888/predict",
                    "3": "http://localhost:6006/predict"
                },
                "model": {
                    "url": "http://localhost:8068/predict",
                    "time_out": 10
                },
                "nns": {
                    "time_out": 10,
                    "ai_threshold": 0.2,
                    "human_threshold": 0.1
                }
            }
        }

        return value

    def get_model_timeout(self):
        try:
            return self.value['application']['model']['time_out']
        except Exception as e:
            print(e)
            traceback.print_exc()
        return 10

    def get_nns_timeout(self):
        try:
            return self.value['application']['nns']['time_out']
        except Exception as e:
            print(e)
            traceback.print_exc()
        return 10

    def get_nns_ai_threshold(self):
        try:
            return self.value['application']['nns']['ai_threshold']
        except Exception as e:
            print(e)
            traceback.print_exc()
        return 0.2

    def get_nns_hu_threshold(self):
        try:
            return self.value['application']['nns']['human_threshold']
        except Exception as e:
            print(e)
            traceback.print_exc()
        return 0.1

    def get_nns_server_url(self, hotkey):
        try:
            pile_file_number = self.value['application']['validator_pile'][hotkey]
            url = self.value['application']['pile_server'][str(pile_file_number)]
            return url
        except Exception as e:
            print(e)
            traceback.print_exc()
        return None

    def get_model_server_url(self):
        try:
            return self.value['application']['model']['url']
        except Exception as e:
            print(e)
            traceback.print_exc()
        return None

    def load_app_config(self):
        print("start load_app_config")
        try:
            with open(self.config_path, 'r') as file:
                self.value = json.load(file)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.value = self.default_app_config()
        finally:
            print("finish load_app_config " + str(self.value))


if __name__ == '__main__':
    app_config = AppConfig('/Users/nannan/IdeaProjects/bittensor/sn32-wolfteam/application.json')
    print(app_config)
    print(app_config.value)
    print('get_model_timeout', app_config.get_model_timeout())
    print('get_nns_timeout', app_config.get_nns_timeout())
    print('get_nns_ai_threshold', app_config.get_nns_ai_threshold())
    print('get_nns_hu_threshold', app_config.get_nns_hu_threshold())
    print('get_nns_server_url', app_config.get_nns_server_url("5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3"))
    print('get_model_server_url', app_config.get_model_server_url())


    while True:
        ...
