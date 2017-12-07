#pragma once

#include <string>

enum class ErrorType {UNKNOWN_ERROR, WARNING, FILE_NOT_FOUND, USER_ERROR};

class Error
{
    public:
        static void add(ErrorType _type, std::string _description);
        static bool check();

    private:
        Error();
        ~Error();

    /// Attributes (private)
        static bool error;

    /// Methods (private)
        static std::string getTitle(ErrorType _type);
        static int getIcon(ErrorType _type);
};
