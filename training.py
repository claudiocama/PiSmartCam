import os, time, pickle
import face_recognition


class Training():

    def __init__(self, dataset_directory="static/dataset", models_directory="static/models", model_name = "Default"):
        self.model_name = model_name
        self.dataset_directory = dataset_directory
        self.models_directory = models_directory
        self.known_face_encodings = []
        self.known_face_names = []
        self.dictionary = {}

    def train(self):
        start = time.time()
        count_person = 1
        count_image = 1
        for person in os.listdir(self.dataset_directory+"/"):
            for image in os.listdir(self.dataset_directory+"/"+person):
                print("[INFO]Training image {}/{} of person {}/{} ({})".format(count_image,len(os.listdir(self.dataset_directory+"/"+person)),count_person,len(os.listdir(self.dataset_directory+"/")),person))
                person_image = face_recognition.load_image_file(self.dataset_directory+"/"+person+"/"+image)
                if len(face_recognition.face_encodings(person_image)) > 0:
                    self.known_face_encodings.append(face_recognition.face_encodings(person_image)[0])
                    self.known_face_names.append(person)
                else:
                    print("[WARNING]No face found in this image")
                count_image +=1
            count_image = 1
            count_person += 1
        print("[SUCCESS]Completed in {} seconds".format(str(time.time() - start)))

    def save(self, name="facesDB.p"):
        self.dictionary = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        path = self.models_directory + "/" + name
        pickle.dump(self.dictionary, open(path, "wb"))
        print("[INFO]Model {} saved correctly".format(name))

    def load(self, name="facesDB.p"):
        self.dictionary = pickle.load(open("static/models/"+name, "rb"))
        self.known_face_encodings = self.dictionary["encodings"]
        self.known_face_names = self.dictionary["names"]
        self.model_name = name
        print("[INFO]Model {} loaded correctly".format(name))

    def empty_model(self):
        self.known_face_encodings = []
        self.known_face_names = []
        print("[INFO]Model is now empty")

    def get_encodings(self):
        return self.known_face_encodings

    def get_names(self):
        return self.known_face_names

    def get_dataset_directory(self):
        return self.dataset_directory

    def get_models_directory(self):
        return self.models_directory

    def get_number_of_images(self):
        return len(self.known_face_encodings)

    def get_model_name(self):
        return self.model_name

