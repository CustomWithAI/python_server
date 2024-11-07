dcup-build:
	docker build -t python-server .

dcup-dev:
	docker-compose up --build

dc-down:
	docker-compose down

dc-nuclear:
	- docker stop $(shell docker ps -q)
	- docker kill $(shell docker ps -q)
	- docker rm $(shell docker ps -a -q)
	- docker rmi $(shell docker images -q)
	- docker system prune --all --force --volumes
