# abstraction.py


def abstract_detection(persons):
    flags = {
        "person": False,
        "helmet": False,
        "harness": False,
        "person_box": None
    }

    if not persons:
        return flags
    
    person = persons[0]
    flags["person"] = True
    flags["helmet"] = person["helmet"]  
    flags["harness"] = person["harness"]
    flags["person_box"] = person["bbox"]


    return flags