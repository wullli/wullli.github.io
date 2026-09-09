.PHONY: build serve clean

build:
	node build.js

serve: build
	python3 -m http.server -d _site 8000

clean:
	rm -rf _site
