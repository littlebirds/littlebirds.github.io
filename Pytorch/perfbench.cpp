#include <torch/torch.h>
#include <torch/script.h>
// #include <torchvision/vision.h>
#include <ATen/ATen.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <dirent.h>

torch::Tensor read_data(std::string &loc)
{
  // Read Image from the location of image
  cv::Mat img = cv::imread(loc, 0);
  cv::resize(img, img, cv::Size(224, 224), cv::INTER_LINEAR);
  // std::cout << "Sizes: " << img.size() << std::endl;
  torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kByte);
  img_tensor = img_tensor.permute({2, 0, 1}); // Channels x Height x Width

  return img_tensor.clone();
};

torch::Tensor read_label(int label)
{
  torch::Tensor label_tensor = torch::full({1}, label);
  return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> &list_images)
{
  using namespace std;
  cout << "Reading images..." << endl;
  vector<torch::Tensor> states;
  for (std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it)
  {
    // cout << "Location being read: " << *it << endl;
    torch::Tensor img = read_data(*it);
    states.push_back(img);
  }
  cout << "Reading and Processing images done!" << endl;
  return states;
}

std::vector<torch::Tensor> process_labels(std::vector<int> &list_labels)
{
  std::cout << "Reading labels..." << std::endl;
  std::vector<torch::Tensor> labels;
  for (auto it = list_labels.begin(); it != list_labels.end(); ++it)
  {
    torch::Tensor label = read_label(*it);
    labels.push_back(label);
  }
  std::cout << "Labels reading done!" << std::endl;
  return labels;
}

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
  // Declare 2 vectors of tensors for images and labels
  std::vector<torch::Tensor> images, labels;

public:
  // Constructor
  CustomDataset(std::vector<std::string> &list_images, std::vector<int> &list_labels)
  {
    images = process_images(list_images);
    labels = process_labels(list_labels);
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override
  {
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
  };

  // Return the length of data
  torch::optional<size_t> size() const override
  {
    return labels.size();
  };
};

/* This function returns a pair of vector of images paths (strings) and labels (integers) */
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> &folders_name)
{
  using namespace std;
  vector<string> list_images;
  vector<int> list_labels;
  int label = 0;
  for (auto const &value : folders_name)
  {
    string base_name = value + "/";
    cout << "Reading from: " << base_name << endl;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(base_name.c_str())) != NULL)
    {
      while ((ent = readdir(dir)) != NULL)
      {
        string filename = ent->d_name;
        if (filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg")
        {
          //cout << base_name + ent->d_name << endl;
          // cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
          list_images.push_back(base_name + ent->d_name);
          list_labels.push_back(label);
        }
      }
      closedir(dir);
    }
    else
    {
      cout << "Could not open directory" << endl;
      // return EXIT_FAILURE;
    }
    label += 1;
  }
  return std::make_pair(list_images, list_labels);
}

int main(int argc, const char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  torch::Device device(torch::kCUDA);

  torch::jit::script::Module module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    module.to(device);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }
  // data loader
  using namespace std;
  std::string rootDir("/home/gangzhi/Documents/DeepLearn/cats_dogs/test/");
  vector<string> folders_name;
  folders_name.push_back(rootDir + "cat");
  folders_name.push_back(rootDir + "dog");
  std::vector<std::string> list_images;
  std::vector<int> list_labels;
  std::tie(list_images, list_labels) = load_data_from_folder(folders_name);

  auto custom_dataset = CustomDataset(list_images, list_labels)
                            .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                            .map(torch::data::transforms::Stack<>()); // batch samplings into 1 tensor
  // Generate a data loader.
  //auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  //  std::move(custom_dataset),
  //  16 /*batch-size*/);
  auto data_loader = torch::data::make_data_loader(std::move(custom_dataset),
                                                   torch::data::DataLoaderOptions().batch_size(16).workers(6));
  float Loss = 0, Acc = 0;
  torch::NoGradGuard no_grad;
  auto start = std::chrono::high_resolution_clock::now();
  for (auto &batch : *data_loader)
  {
    auto data = batch.data.to(device);
    auto targets = batch.target.view({-1});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(data);
    auto output = module.forward(inputs).toTensor();
    //auto loss = torch::nll_loss(output, targets);
    //auto acc = output.argmax(1).eq(targets).sum();

    //Loss += loss.template item<float>();
    //Acc += acc.template item<float>();
    //  _, predicted = torch.max(outputs.data, 1)
    //  total += labels.size(0)
    //  correct += (predicted == labels).sum().item()
  }
  auto stop = std::chrono::high_resolution_clock::now();
  using namespace std::chrono;
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Total time:" << duration.count() << endl;
  //print('correct: {:d}  total: {:d}'.format(correct, total))
  // print('accuracy = {:f}'.format(correct / total))
}
