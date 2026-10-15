.PHONY: build serve dev clean

node_modules: package.json
	npm install
	touch node_modules

build: node_modules
	node build.js

serve: build
	python3 -m http.server -d _site 8000

dev: node_modules
	node serve.js

clean:
	rm -rf _site
