#include "SafeTensorLoader.hpp"
#include <iostream>

int wmain(int argc, wchar_t** wideArgs) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        auto utf8 = std::filesystem::path(wideArgs[i]).u8string();
        arguments.emplace_back(utf8.begin(), utf8.end());
    }
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    try {
        if (argc != 4 && argc != 5) throw std::runtime_error("Expected file, tensor name, output file, optional base directory");
        auto tensor = PyTorch::LoadSafeTensor(PyTorch::ResolveTensorFile(argv[1], argc == 5 ? argv[4] : ""), argv[2]);
        const std::string outputName(argv[3]);
        std::ofstream output(std::filesystem::path(std::u8string(outputName.begin(), outputName.end())), std::ios::binary);
        output.write(static_cast<const char*>(tensor.const_data_ptr()), tensor.nbytes());
        if (!output) throw std::runtime_error("Cannot write output");
        std::cout << c10::toString(tensor.scalar_type()) << '\n';
        for (auto dimension : tensor.sizes()) std::cout << dimension << ' ';
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
