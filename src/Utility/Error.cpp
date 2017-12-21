#include "Error.h"
#include "windows.h"

bool Error::error = false;

Error::Error()
{ }

Error::~Error()
{ }

void Error::add(ErrorType _type, std::string _description)
{
    MessageBox(nullptr, _description.c_str(), getTitle(_type).c_str(), getIcon(_type));

    error = true;
}

bool Error::check()
{
    if (!error)
        return false;

    error = false;

    int r = MessageBox(nullptr, "One or more errors occured.\nDo you really want to continue ?",
                       "MinGE: loading error", MB_ICONQUESTION | MB_YESNO | MB_DEFBUTTON2);

    return (r == IDNO);
}

std::string Error::getTitle(ErrorType _type)
{
    switch (_type)
    {
        case ErrorType::UNKNOWN_ERROR:
            return "Error";
        break;
        case ErrorType::WARNING:
            return "Warning";
        break;
        case ErrorType::FILE_NOT_FOUND:
            return "File not found";
        break;
        case ErrorType::USER_ERROR:
            return "User error";
        break;

        default:
            return "Unknown error";
        break;
    }
}

int Error::getIcon(ErrorType _type)
{
    switch (_type)
    {
        case ErrorType::UNKNOWN_ERROR:
            return MB_ICONERROR;
        break;
        case ErrorType::WARNING:
            return MB_ICONWARNING;
        break;
        case ErrorType::FILE_NOT_FOUND:
            return MB_ICONINFORMATION;
        break;
        case ErrorType::USER_ERROR:
            return MB_ICONERROR;
        break;

        default:
            return MB_ICONERROR;
        break;
    }
}
