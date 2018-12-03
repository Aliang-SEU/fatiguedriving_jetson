#ifndef MESSAGE
#define MESSAGE

#include<string>

enum MessageType{
    NORMAL_DRIVING,         //正常驾驶
    LIGHT_FATIGUE_DRIVING,  //轻度疲劳
    SEVERE_FATIGUE_DRIVING  //严重疲劳
};

class Message {
public:

    Message(){}
    Message(std::string mes){ this->message = mes;}
    Message(std::string mes, MessageType type) {
        this->message = mes;
        this->messageType = type;
    }
    void setMessage(std::string mes) {
        this->message = mes;
    }

    std::string getMessage() {
        return this->message;
    }

    void setMessageType(MessageType type) {
        this->messageType = type;
    }

    Message getMessage() {
        return this->messageType;
    }

private:
    std::string message;
    MessageType messageType;
}

#endif // MESSAGE

