from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)


# python3 -m gunicorn -w 4 -b 0.0.0.0:8080 --reload run:app