import os, time, pickle
import face_recognition


class Training():

    def __init__(self, directory="dataset"):
        self.directory = directory
        self.known_face_encodings = []
        self.known_face_names = []
        self.dictionary = {}

    def train(self):
        start = time.time()
        count_person = 1
        count_image = 1
        for person in os.listdir(self.directory+"/"):
            for image in os.listdir(self.directory+"/"+person):
                print("[INFO]Training image {}/{} of person {}/{} ({})".format(count_image,len(os.listdir(self.directory+"/"+person)),count_person,len(os.listdir(self.directory+"/")),person))
                person_image = face_recognition.load_image_file(self.directory+"/"+person+"/"+image)
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
        pickle.dump(self.dictionary, open(name, "wb"))

    def load(self, name="facesDB.p"):
        self.dictionary = pickle.load(open(name, "rb"))
        self.known_face_encodings = self.dictionary["encodings"]
        self.known_face_names = self.dictionary["names"]

    def get_encodings(self):
        return self.known_face_encodings

    def get_names(self):
        return self.known_face_names
