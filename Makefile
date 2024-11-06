dcup-build:
	docker build -t python-server .

dcup-dev:
	docker run -p 8000:8000 python-server

dc-down:
	docker-compose down

dc-nuclear:
	- docker stop $$(docker ps -a -q)
	- docker kill $$(docker ps -q)
	- docker rm $$(docker ps -a -q)
	- docker rmi $$(docker images -q)
	- docker system prune --all --force --volumes