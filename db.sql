DROP DATABASE IF EXISTS text_news;
CREATE DATABASE text_news;
USE text_news;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(225),
    email VARCHAR(225),
    password VARCHAR(225)
);
