#include <chrono>
#include <iostream>

#include "Network.hpp"
#include <SFML/Graphics.hpp>

#include <array>
#include <cmath>

using Net = Network<Sigmoid, 4,16,16,6>;

struct SFMLObject {
    sf::CircleShape shape;
    SFMLObject(float x, float y, float radius, sf::Color color)
        : shape(radius) {
        shape.setFillColor(color);
        shape.setPosition({x, y});
    }
    float getX() const { return shape.getPosition().x; }
    float getY() const { return shape.getPosition().y; }
    float getRadius() const { return shape.getRadius(); }

    sf::Vector2f getCenter() const {
        const float r = shape.getRadius();
        return {shape.getPosition().x + r, shape.getPosition().y + r};
    }

    void setPosition(float x, float y) { shape.setPosition({x, y}); }
    void setPosition(sf::Vector2f position) { shape.setPosition(position); }
    void draw(sf::RenderWindow& w) const { w.draw(shape); }
};


enum class AgentAction {
    moveRight,
    moveLeft,
    moveForward,
    moveBackward,
    turnLeft,
    turnRight
};

struct Agent : SFMLObject {
    Net   net{};
    float theta = 0.f;

    using SFMLObject::SFMLObject;

    static std::pair<float,float> worldToBody(float dx, float dy, float theta){
        const float c = std::cos(theta), s = std::sin(theta);
        float xb =  c*dx + s*dy;
        float yb = -s*dx + c*dy;
        return {xb, yb};
    }

    static std::pair<float,float> bodyToWorld(float x_b, float y_b, float theta){
        const float c = std::cos(theta), s = std::sin(theta);
        float x =  c*x_b - s*y_b;
        float y =  s*x_b + c*y_b;
        return {x, y};
    }

    void update(const SFMLObject& obj, float dt /* seconds */) {
        const sf::Vector2f myC   = this->getCenter();
        const sf::Vector2f objC  = obj.getCenter();
        float dx = objC.x - myC.x;
        float dy = objC.y - myC.y;

        auto [xb, yb] = worldToBody(dx, dy, theta);
        float r   = std::hypot(xb, yb);
        float phi = std::atan2(yb, xb);
        float cph = std::cos(phi);
        float sph = std::sin(phi);

        std::array input{ r, cph, sph, 1.0f };

        auto action = net.forward(input);

        float strafe = 0.f;
        float fwd    = 0.f;
        float dtheta = 0.f;

        if (action[static_cast<int>(AgentAction::moveForward)]  > 0.5f) fwd    += 1.f;
        if (action[static_cast<int>(AgentAction::moveBackward)] > 0.5f) fwd    -= 1.f;
        if (action[static_cast<int>(AgentAction::moveRight)]    > 0.5f) strafe += 1.f;
        if (action[static_cast<int>(AgentAction::moveLeft)]     > 0.5f) strafe -= 1.f;
        if (action[static_cast<int>(AgentAction::turnLeft)]     > 0.5f) dtheta += 1.f;
        if (action[static_cast<int>(AgentAction::turnRight)]    > 0.5f) dtheta -= 1.f;

        float len = std::hypot(strafe, fwd);
        if (len > 1e-6f) { strafe /= len; fwd /= len; }

        constexpr float MOVE_SPEED = 180.f;
        constexpr float TURN_SPEED = 2.5f;

        theta += dtheta * TURN_SPEED * dt;

        auto [vx, vy] = bodyToWorld(fwd,-strafe, theta);

        float newX = getX() + vx * MOVE_SPEED * dt;
        float newY = getY() + vy * MOVE_SPEED * dt;
        setPosition(newX, newY);
    }
};

static void respawnApple(SFMLObject& apple, std::mt19937& rng) {
    const float r = apple.getRadius();
    const float maxX = static_cast<float>(800) - r;
    const float maxY = static_cast<float>(600) - r;
    std::uniform_real_distribution<float> distX(r, maxX);
    std::uniform_real_distribution<float> distY(r, maxY);

    const sf::Vector2f pos(distX(rng), distY(rng));
    apple.setPosition(pos);
}


int main() {
    constexpr int   N_AGENTS = 10000;
    constexpr int   simTicks = 300;
    constexpr float dt       = 1.0f / 60.0f;
    sf::Vector2f agentStart = {400, 300};
    sf::Vector2f appleStart = {300, 200};

    std::random_device rd;
    std::mt19937 masterRng(rd());

    std::vector<Agent> agents;
    agents.reserve(N_AGENTS);
    for (int i = 0; i < N_AGENTS; ++i) {
        agents.emplace_back(agentStart.x, agentStart.y, 10, sf::Color::White);
    }

    double bestScore = -std::numeric_limits<double>::infinity();
    std::size_t bestIdx = 0;

    std::vector<sf::Vector2f> bestAppleTimeline;
    std::vector<sf::Vector2f> bestAgentTimeline;
    bestAppleTimeline.reserve(simTicks);
    bestAgentTimeline.reserve(simTicks);

    for (std::size_t i = 0; i < agents.size(); ++i) {
        Agent a = agents[i];
        a.setPosition(agentStart);
        a.theta = 0.f;

        SFMLObject apple(appleStart.x, appleStart.y, 10, sf::Color::Red);

        std::mt19937 rng(masterRng());

        int score = 0;
        std::vector<sf::Vector2f> appleTimeline(simTicks);
        std::vector<sf::Vector2f> agentTimeline(simTicks);

        for (int t = 0; t < simTicks; ++t) {
            a.update(apple, dt);

            const sf::Vector2f aC  = a.getCenter();
            const sf::Vector2f apC = apple.getCenter();
            const float dx = aC.x - apC.x;
            const float dy = aC.y - apC.y;
            const float rr = a.getRadius() + apple.getRadius();

            if (dx * dx + dy * dy <= rr * rr) {
                ++score;
                respawnApple(apple, rng);
            }

            agentTimeline[t] = a.shape.getPosition();
            appleTimeline[t] = apple.shape.getPosition();
        }

        if (score > bestScore) {
            bestScore = static_cast<double>(score);
            bestIdx   = i;
            bestAppleTimeline = std::move(appleTimeline);
            bestAgentTimeline = std::move(agentTimeline);
        }
    }

    std::cout << "Best score: " << bestScore << std::endl;

    sf::RenderWindow window(sf::VideoMode({800, 600}), "Best Agent Replay");
    window.setFramerateLimit(60);

    Agent bestAgent = agents[bestIdx];
    SFMLObject apple(0, 0, 10, sf::Color::Red);

    std::size_t tick = 0;
    while (window.isOpen()) {
        while (auto evt = window.pollEvent()) {
            if (!evt) break;
            if (evt->is<sf::Event::Closed>()) window.close();
        }

        if (tick < bestAppleTimeline.size() && tick < bestAgentTimeline.size()) {
            bestAgent.shape.setPosition(bestAgentTimeline[tick]);
            apple.shape.setPosition(bestAppleTimeline[tick]);
            ++tick;
        } else {
            window.close();
            continue;
        }

        window.clear(sf::Color::Black);
        window.draw(bestAgent.shape);
        window.draw(apple.shape);
        window.display();
    }

    return 0;
}
