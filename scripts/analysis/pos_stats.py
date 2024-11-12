from eagle.phrase.pos import POSParser


def main():
    parser = POSParser()
    result = parser("This is a test sentence.")
    stop = 1


if __name__ == "__main__":
    main()
