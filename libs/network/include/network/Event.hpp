//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_EVENT_HPP
#define NEUVISYS_DV_EVENT_HPP

#include <cstdint>

class Event {
    uint64_t m_timestamp;
    uint16_t m_x;
    uint16_t m_y;
    bool m_polarity;
    uint16_t m_camera;
    uint16_t m_synapse;
    bool m_over;

public:
    Event() = default;

    inline Event(uint64_t timestamp, uint16_t x, uint16_t y, bool polarity) : m_timestamp(timestamp), m_x(x), m_y(y), m_polarity(polarity),
                                                                              m_camera(0) { m_synapse = 0; m_over = false; }

    inline Event(uint64_t timestamp, uint16_t x, uint16_t y, bool polarity, uint16_t camera) : m_timestamp(timestamp), m_x(x), m_y(y),
                                                                                               m_polarity(polarity),
                                                                                               m_camera(camera) { m_synapse = 0; m_over = false; }

    inline Event(uint64_t timestamp, uint16_t x, uint16_t y, bool polarity, uint16_t camera, uint16_t synapse) : m_timestamp(timestamp), m_x(x),
                                                                                                                 m_y(y), m_polarity(polarity),
                                                                                                                 m_camera(camera),
                                                                                                                 m_synapse(synapse) { m_over = false; }

    inline Event(uint64_t timestamp, uint16_t x, uint16_t y, bool polarity, uint16_t camera, uint16_t synapse, bool over) : m_timestamp(timestamp), m_x(x),
                                                                                                                 m_y(y), m_polarity(polarity),
                                                                                                                 m_camera(camera),
                                                                                                                 m_synapse(synapse),
                                                                                                                 m_over(over) {}

    [[nodiscard]] inline uint64_t timestamp() const { return m_timestamp; }

    [[nodiscard]] inline uint16_t x() const { return m_x; }

    [[nodiscard]] inline uint16_t y() const { return m_y; }

    [[nodiscard]] inline bool polarity() const { return m_polarity; }

    [[nodiscard]] inline uint16_t synapse() const { return m_synapse; }

    [[nodiscard]] inline uint16_t camera() const { return m_camera; }

    [[nodiscard]] inline uint16_t over() const { return m_over; }

    bool operator<(const Event &event) const {
        return m_timestamp < event.m_timestamp;
    }

    void operator=(Event event) {
        m_x = event.x();
        m_y = event.y();
        m_polarity = event.polarity();
        m_timestamp = event.timestamp();
        m_camera = event.camera();
        m_synapse = event.synapse();
        m_over = event.over();
    }
};

class NeuronEvent {
    uint64_t m_timestamp{};
    uint16_t m_x;
    uint16_t m_y;
    uint16_t m_z;
    uint32_t m_id{};
    uint32_t m_layer{};

public:
    NeuronEvent() = default;

    NeuronEvent(uint64_t timestamp, uint16_t x, uint16_t y, uint16_t z, uint32_t id, uint32_t layer) : m_timestamp(timestamp), m_x(x), m_y(y), m_z(z), m_id(id), m_layer(layer) {}

    [[nodiscard]] uint64_t timestamp() const { return m_timestamp; }

    [[nodiscard]] uint32_t x() const { return m_x; }

    [[nodiscard]] uint32_t y() const { return m_y; }

    [[nodiscard]] uint32_t z() const { return m_z; }

    [[nodiscard]] uint32_t id() const { return m_id; }

    [[nodiscard]] uint32_t layer() const { return m_layer; }

    bool operator<(const NeuronEvent &event) const {
        return m_timestamp < event.m_timestamp;
    }
};

#endif //NEUVISYS_DV_EVENT_HPP
