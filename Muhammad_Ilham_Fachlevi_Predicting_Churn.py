{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "8rApEjnhBUqz"
      ],
      "toc_visible": true,
      "authorship_tag": "ABX9TyOlADwSfitRjjrVZbiwy57f",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ilhamfachlevi/Data-Science-Project-5-Predicting-Churn-Probability-using-Machine-Learning-Models/blob/main/Muhammad_Ilham_Fachlevi_Predicting_Churn.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Kaggle Installation"
      ],
      "metadata": {
        "id": "fWq1PdPKmOoX"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O6mXS6Rzyc-r"
      },
      "outputs": [],
      "source": [
        "# install kaggle\n",
        "!pip install -q kaggle"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 90
        },
        "id": "jp-SdxLhzMwm",
        "outputId": "7479e6a7-9c37-4867-f3e3-459765abbde0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-36667c1b-49a0-47a7-b5e7-386d5d3c626d\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-36667c1b-49a0-47a7-b5e7-386d5d3c626d\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving kaggle.json to kaggle (2).json\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'kaggle.json': b'{\"username\":\"milhamfachlevi\",\"key\":\"b7c866182fbd6d70be22a6aff25d66b1\"}'}"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# create a kaggle folder\n",
        "!mkdir ~/.kaggle"
      ],
      "metadata": {
        "id": "9af2_TOd05cf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1ed8688a-310e-4e7b-f5e6-e6e9a4e6a155"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mkdir: cannot create directory ‘/root/.kaggle’: File exists\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# copy the kaggle.json to folder created  \n",
        "!cp kaggle.json ~/.kaggle/"
      ],
      "metadata": {
        "id": "mEkdwWqM1Kzy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# permisson for the json to act\n",
        "!chmod 600 ~/.kaggle/kaggle.json"
      ],
      "metadata": {
        "id": "hgoDGasG1OFa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# to list all avalaible datasets in kaggle\n",
        "!kaggle datasets list"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R_gf4LYI3UMv",
        "outputId": "a55cc5d2-c23b-4f7f-f8b8-807f8e66a075"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ref                                                             title                                            size  lastUpdated          downloadCount  voteCount  usabilityRating  \n",
            "--------------------------------------------------------------  ----------------------------------------------  -----  -------------------  -------------  ---------  ---------------  \n",
            "akshaydattatraykhare/diabetes-dataset                           Diabetes Dataset                                  9KB  2022-10-06 08:55:25          10521        337  1.0              \n",
            "whenamancodes/covid-19-coronavirus-pandemic-dataset             COVID -19 Coronavirus Pandemic Dataset           11KB  2022-09-30 04:05:11           8399        271  1.0              \n",
            "stetsondone/video-game-sales-by-genre                           Video Game Sales by Genre                        12KB  2022-10-31 17:56:01            659         23  1.0              \n",
            "whenamancodes/credit-card-customers-prediction                  Credit Card Customers Prediction                379KB  2022-10-30 13:03:27           1117         34  1.0              \n",
            "whenamancodes/students-performance-in-exams                     Students Performance in Exams                     9KB  2022-09-14 15:14:54          14560        272  1.0              \n",
            "akshaydattatraykhare/data-for-admission-in-the-university       Data for Admission in the University              4KB  2022-10-27 11:05:45           1682         39  1.0              \n",
            "michaelbryantds/electric-vehicle-charging-dataset               Electric Vehicle Charging Dataset                98KB  2022-11-02 01:45:23            531         30  0.9411765        \n",
            "maharshipandya/-spotify-tracks-dataset                          🎹 Spotify Tracks Dataset                          8MB  2022-10-22 14:40:15           2037         67  1.0              \n",
            "hasibalmuzdadid/global-air-pollution-dataset                    Global Air Pollution Dataset                    371KB  2022-11-08 14:43:32           1002         40  1.0              \n",
            "younver/spotify-top-200-dataset                                 spotify top 200 dataset                           9MB  2022-10-29 19:36:07            572         26  1.0              \n",
            "whenamancodes/amazon-reviews-on-women-dresses                   Amazon Reviews on Women Dresses                   3MB  2022-10-29 12:47:06            367         22  1.0              \n",
            "akshaydattatraykhare/car-details-dataset                        Car Details Dataset                              56KB  2022-10-21 06:11:56           2515         43  1.0              \n",
            "whenamancodes/international-football-from-1872-to-2022          International Football from 1872 to 2022        572KB  2022-10-30 13:27:29            615         28  0.9411765        \n",
            "whenamancodes/customer-personality-analysis                     Company's Ideal Customers | Marketing Strategy   62KB  2022-10-30 14:17:42            836         31  1.0              \n",
            "dimitryzub/walmart-coffee-listings-from-500-stores              Walmart Coffee Listings from 500 stores          85KB  2022-10-25 09:20:12           1219         36  1.0              \n",
            "dheerajmukati/most-runs-in-cricket                              Most Runs in International cricket                4KB  2022-10-16 16:49:20            651         30  1.0              \n",
            "thedevastator/food-prices-year-by-year                          Global Food Prices Year By Year                   7KB  2022-10-30 08:49:55            835         29  1.0              \n",
            "saikumartamminana/gold-price-prediction                         Gold Price Prediction                            41KB  2022-10-30 19:07:30            845         24  0.8235294        \n",
            "thedevastator/udemy-courses-revenue-generation-and-course-anal  Udemy Courses                                   429KB  2022-10-17 00:11:53           1797         56  1.0              \n",
            "whenamancodes/adidas-us-retail-products-dataset                 Adidas US Retail Products Dataset               286KB  2022-10-26 15:44:20            663         33  1.0              \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!kaggle datasets download -d olistbr/brazilian-ecommerce"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-6Bv7xfV3me4",
        "outputId": "4e1a8126-ab43-4d30-bbc0-c8e0c21af754"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "brazilian-ecommerce.zip: Skipping, found more recently modified local copy (use --force to force download)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8pWv2AQBGnc5",
        "outputId": "0ce6caf2-c012-4038-df99-96cab1eabf63"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " \u001b[0m\u001b[01;34mbrazilian-ecommerce\u001b[0m/     'kaggle (1).json'   kaggle.json\n",
            " brazilian-ecommerce.zip  'kaggle (2).json'   \u001b[01;34msample_data\u001b[0m/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Import Package"
      ],
      "metadata": {
        "id": "eMOSoYR0mbSe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import glob\n",
        "import os\n",
        "import numpy as numpy\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import zipfile\n",
        "import datetime as dt\n",
        "import statsmodels.api as sm\n",
        "\n",
        "# Model Machine Learning standar untuk perbandingan \n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor\n",
        "from sklearn.ensemble import ExtraTreesClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "\n",
        "from sklearn.metrics import r2_score,accuracy_score,classification_report\n",
        "\n",
        "# Splitting data untuk training/testing\n",
        "from sklearn.model_selection import train_test_split,GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "from sklearn.cluster import KMeans\n",
        "\n",
        "# Importing libraries untuk membuat neural network\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense,LSTM\n",
        "from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "import xgboost as xgb\n",
        "from sklearn.model_selection import KFold, cross_val_score, train_test_split"
      ],
      "metadata": {
        "id": "DaKqvF7Y7Xf3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "with zipfile.ZipFile(\"/content/brazilian-ecommerce.zip\",\"r\") as zip_ref:\n",
        "    zip_ref.extractall(\"brazilian-ecommerce\")"
      ],
      "metadata": {
        "id": "t_ZWc0XFAuEp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "path = r'/content/brazilian-ecommerce' \n",
        "all_files = glob.glob(os.path.join(path , \"*.csv\"))\n",
        "\n",
        "li = []\n",
        "\n",
        "for filename in all_files:\n",
        "    df = pd.read_csv(filename, index_col=None, header=0)\n",
        "    li.append(df)\n",
        "\n",
        "frame = pd.concat(li, axis=0, ignore_index=True)"
      ],
      "metadata": {
        "id": "bL8ZmuoU_XeM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(all_files)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EkfVptSgHsOJ",
        "outputId": "9ffd5cc0-60a7-4405-b73b-d9aa53506777"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['/content/brazilian-ecommerce/olist_orders_dataset.csv', '/content/brazilian-ecommerce/olist_order_reviews_dataset.csv', '/content/brazilian-ecommerce/olist_sellers_dataset.csv', '/content/brazilian-ecommerce/olist_geolocation_dataset.csv', '/content/brazilian-ecommerce/product_category_name_translation.csv', '/content/brazilian-ecommerce/olist_order_payments_dataset.csv', '/content/brazilian-ecommerce/olist_products_dataset.csv', '/content/brazilian-ecommerce/olist_customers_dataset.csv', '/content/brazilian-ecommerce/olist_order_items_dataset.csv']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Initializing Variables"
      ],
      "metadata": {
        "id": "WCkr-u7DnAAF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "customers = pd.read_csv('/content/brazilian-ecommerce/olist_customers_dataset.csv')\n",
        "geolocation = pd.read_csv('/content/brazilian-ecommerce/olist_geolocation_dataset.csv')\n",
        "items = pd.read_csv('/content/brazilian-ecommerce/olist_order_items_dataset.csv')\n",
        "payments = pd.read_csv('/content/brazilian-ecommerce/olist_order_payments_dataset.csv')\n",
        "reviews = pd.read_csv('/content/brazilian-ecommerce/olist_order_reviews_dataset.csv')\n",
        "orders = pd.read_csv('/content/brazilian-ecommerce/olist_orders_dataset.csv')\n",
        "products = pd.read_csv('/content/brazilian-ecommerce/olist_products_dataset.csv')\n",
        "name = pd.read_csv('/content/brazilian-ecommerce/product_category_name_translation.csv')\n",
        "sellers = pd.read_csv('/content/brazilian-ecommerce/olist_sellers_dataset.csv')"
      ],
      "metadata": {
        "id": "kU96_NUbHxM5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = {\n",
        "    'customers': customers,\n",
        "    'geolocation': geolocation,\n",
        "    'items': items,\n",
        "    'paymens':payments,\n",
        "    'reviews':reviews,\n",
        "    'orders':orders,\n",
        "    'products':products,\n",
        "    'name':name,\n",
        "    'sellers':sellers\n",
        "}"
      ],
      "metadata": {
        "id": "cqc_sINh4dNJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "customers.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "K71GelIAPxQq",
        "outputId": "558becfb-e10f-4de5-d2fb-2e486ef7c8cf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 99441 entries, 0 to 99440\n",
            "Data columns (total 5 columns):\n",
            " #   Column                    Non-Null Count  Dtype \n",
            "---  ------                    --------------  ----- \n",
            " 0   customer_id               99441 non-null  object\n",
            " 1   customer_unique_id        99441 non-null  object\n",
            " 2   customer_zip_code_prefix  99441 non-null  int64 \n",
            " 3   customer_city             99441 non-null  object\n",
            " 4   customer_state            99441 non-null  object\n",
            "dtypes: int64(1), object(4)\n",
            "memory usage: 3.8+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "geolocation.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LTmsyQSjP0Dv",
        "outputId": "a334ab8e-d6c0-4d0c-cc87-e02287ca5775"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1000163 entries, 0 to 1000162\n",
            "Data columns (total 5 columns):\n",
            " #   Column                       Non-Null Count    Dtype  \n",
            "---  ------                       --------------    -----  \n",
            " 0   geolocation_zip_code_prefix  1000163 non-null  int64  \n",
            " 1   geolocation_lat              1000163 non-null  float64\n",
            " 2   geolocation_lng              1000163 non-null  float64\n",
            " 3   geolocation_city             1000163 non-null  object \n",
            " 4   geolocation_state            1000163 non-null  object \n",
            "dtypes: float64(2), int64(1), object(2)\n",
            "memory usage: 38.2+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "items.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LixZDkm-P3qe",
        "outputId": "5f8e85e0-6bb4-414d-ed1b-65521d9fdf5a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 112650 entries, 0 to 112649\n",
            "Data columns (total 7 columns):\n",
            " #   Column               Non-Null Count   Dtype  \n",
            "---  ------               --------------   -----  \n",
            " 0   order_id             112650 non-null  object \n",
            " 1   order_item_id        112650 non-null  int64  \n",
            " 2   product_id           112650 non-null  object \n",
            " 3   seller_id            112650 non-null  object \n",
            " 4   shipping_limit_date  112650 non-null  object \n",
            " 5   price                112650 non-null  float64\n",
            " 6   freight_value        112650 non-null  float64\n",
            "dtypes: float64(2), int64(1), object(4)\n",
            "memory usage: 6.0+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "payments.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VcSfzO2cP6Mg",
        "outputId": "16ed85d6-fdeb-45a5-ac12-c0a6ae88da6b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 103886 entries, 0 to 103885\n",
            "Data columns (total 5 columns):\n",
            " #   Column                Non-Null Count   Dtype  \n",
            "---  ------                --------------   -----  \n",
            " 0   order_id              103886 non-null  object \n",
            " 1   payment_sequential    103886 non-null  int64  \n",
            " 2   payment_type          103886 non-null  object \n",
            " 3   payment_installments  103886 non-null  int64  \n",
            " 4   payment_value         103886 non-null  float64\n",
            "dtypes: float64(1), int64(2), object(2)\n",
            "memory usage: 4.0+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "reviews.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Eh5h0rYUP_xR",
        "outputId": "767dd24b-b3cc-4cf9-d06b-e60d53ceecc6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 99224 entries, 0 to 99223\n",
            "Data columns (total 7 columns):\n",
            " #   Column                   Non-Null Count  Dtype \n",
            "---  ------                   --------------  ----- \n",
            " 0   review_id                99224 non-null  object\n",
            " 1   order_id                 99224 non-null  object\n",
            " 2   review_score             99224 non-null  int64 \n",
            " 3   review_comment_title     11568 non-null  object\n",
            " 4   review_comment_message   40977 non-null  object\n",
            " 5   review_creation_date     99224 non-null  object\n",
            " 6   review_answer_timestamp  99224 non-null  object\n",
            "dtypes: int64(1), object(6)\n",
            "memory usage: 5.3+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "orders.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "12KQ133JQC2K",
        "outputId": "89009667-fff5-47c1-8e6b-8e997d8f17f1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 99441 entries, 0 to 99440\n",
            "Data columns (total 8 columns):\n",
            " #   Column                         Non-Null Count  Dtype \n",
            "---  ------                         --------------  ----- \n",
            " 0   order_id                       99441 non-null  object\n",
            " 1   customer_id                    99441 non-null  object\n",
            " 2   order_status                   99441 non-null  object\n",
            " 3   order_purchase_timestamp       99441 non-null  object\n",
            " 4   order_approved_at              99281 non-null  object\n",
            " 5   order_delivered_carrier_date   97658 non-null  object\n",
            " 6   order_delivered_customer_date  96476 non-null  object\n",
            " 7   order_estimated_delivery_date  99441 non-null  object\n",
            "dtypes: object(8)\n",
            "memory usage: 6.1+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "products.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cC9VEEYzQKME",
        "outputId": "8244a7cc-d96b-4214-a1ac-502b10a8b241"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 32951 entries, 0 to 32950\n",
            "Data columns (total 9 columns):\n",
            " #   Column                      Non-Null Count  Dtype  \n",
            "---  ------                      --------------  -----  \n",
            " 0   product_id                  32951 non-null  object \n",
            " 1   product_category_name       32341 non-null  object \n",
            " 2   product_name_lenght         32341 non-null  float64\n",
            " 3   product_description_lenght  32341 non-null  float64\n",
            " 4   product_photos_qty          32341 non-null  float64\n",
            " 5   product_weight_g            32949 non-null  float64\n",
            " 6   product_length_cm           32949 non-null  float64\n",
            " 7   product_height_cm           32949 non-null  float64\n",
            " 8   product_width_cm            32949 non-null  float64\n",
            "dtypes: float64(7), object(2)\n",
            "memory usage: 2.3+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sellers.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MIowPUEzQNdm",
        "outputId": "0b162662-3ca7-44b9-e3cf-f1e996538db1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 3095 entries, 0 to 3094\n",
            "Data columns (total 4 columns):\n",
            " #   Column                  Non-Null Count  Dtype \n",
            "---  ------                  --------------  ----- \n",
            " 0   seller_id               3095 non-null   object\n",
            " 1   seller_zip_code_prefix  3095 non-null   int64 \n",
            " 2   seller_city             3095 non-null   object\n",
            " 3   seller_state            3095 non-null   object\n",
            "dtypes: int64(1), object(3)\n",
            "memory usage: 96.8+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for name, df in data.items():\n",
        "  key_count = [col for col in df.columns if 'id' in col or 'code' in col]\n",
        "  print(f'{name}: {len(key_count)} PKs pr FKs')\n",
        "  print(f'{key_count}\\n')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z3kSHYCOSp85",
        "outputId": "4fb1edf5-4ece-4957-883e-09514fd0f70d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "customers: 3 PKs pr FKs\n",
            "['customer_id', 'customer_unique_id', 'customer_zip_code_prefix']\n",
            "\n",
            "geolocation: 1 PKs pr FKs\n",
            "['geolocation_zip_code_prefix']\n",
            "\n",
            "items: 4 PKs pr FKs\n",
            "['order_id', 'order_item_id', 'product_id', 'seller_id']\n",
            "\n",
            "paymens: 1 PKs pr FKs\n",
            "['order_id']\n",
            "\n",
            "reviews: 2 PKs pr FKs\n",
            "['review_id', 'order_id']\n",
            "\n",
            "orders: 2 PKs pr FKs\n",
            "['order_id', 'customer_id']\n",
            "\n",
            "products: 2 PKs pr FKs\n",
            "['product_id', 'product_width_cm']\n",
            "\n",
            "name: 0 PKs pr FKs\n",
            "[]\n",
            "\n",
            "sellers: 2 PKs pr FKs\n",
            "['seller_id', 'seller_zip_code_prefix']\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Cleaning"
      ],
      "metadata": {
        "id": "TGGlN_NhRbSe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for name, df in data.items():\n",
        "  if df.isnull().any().any():\n",
        "    print('dataset:',name ,\"\\n\")\n",
        "    print(f'{df.isnull().sum()}\\n')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nUnz9-BPSK7j",
        "outputId": "b5820a67-ede6-4266-a225-923b274ac11f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "dataset: reviews \n",
            "\n",
            "review_id                      0\n",
            "order_id                       0\n",
            "review_score                   0\n",
            "review_comment_title       87656\n",
            "review_comment_message     58247\n",
            "review_creation_date           0\n",
            "review_answer_timestamp        0\n",
            "dtype: int64\n",
            "\n",
            "dataset: orders \n",
            "\n",
            "order_id                            0\n",
            "customer_id                         0\n",
            "order_status                        0\n",
            "order_purchase_timestamp            0\n",
            "order_approved_at                 160\n",
            "order_delivered_carrier_date     1783\n",
            "order_delivered_customer_date    2965\n",
            "order_estimated_delivery_date       0\n",
            "dtype: int64\n",
            "\n",
            "dataset: products \n",
            "\n",
            "product_id                      0\n",
            "product_category_name         610\n",
            "product_name_lenght           610\n",
            "product_description_lenght    610\n",
            "product_photos_qty            610\n",
            "product_weight_g                2\n",
            "product_length_cm               2\n",
            "product_height_cm               2\n",
            "product_width_cm                2\n",
            "dtype: int64\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "orders = orders.dropna(axis='index',subset=['order_approved_at','order_delivered_carrier_date','order_delivered_customer_date'])"
      ],
      "metadata": {
        "id": "SLOJh4PnRdQg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "products = products.dropna(axis='index',subset=['product_category_name', 'product_name_lenght', 'product_description_lenght',\n",
        "                                                'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_width_cm' ])"
      ],
      "metadata": {
        "id": "lFQoFCnG-SDL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "orders.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7ix5opskR91E",
        "outputId": "effe2a3b-f944-4588-ab12-53ada4b7c18d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "order_id                         0\n",
              "customer_id                      0\n",
              "order_status                     0\n",
              "order_purchase_timestamp         0\n",
              "order_approved_at                0\n",
              "order_delivered_carrier_date     0\n",
              "order_delivered_customer_date    0\n",
              "order_estimated_delivery_date    0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "products.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d2ObvgyNtcBN",
        "outputId": "53a316e2-5426-411f-b83f-41f2deeda522"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "product_id                    0\n",
              "product_category_name         0\n",
              "product_name_lenght           0\n",
              "product_description_lenght    0\n",
              "product_photos_qty            0\n",
              "product_weight_g              0\n",
              "product_length_cm             0\n",
              "product_height_cm             0\n",
              "product_width_cm              0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "payments.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fZmrWs6Z70ia",
        "outputId": "8993ddc6-fdf8-4ce4-f186-30c263c23890"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "order_id                0\n",
              "payment_sequential      0\n",
              "payment_type            0\n",
              "payment_installments    0\n",
              "payment_value           0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "reviews['review_comment_title'].fillna(value = 'no comment title', axis = 0, inplace = True)\n",
        "reviews['review_comment_message'].fillna(value = 'no comment message', axis = 0, inplace = True)\n",
        "reviews.dropna(axis = 0, inplace = True)\n",
        "reviews.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 305
        },
        "id": "5hxuvy4n6Mz8",
        "outputId": "0acdee8c-6a39-425f-85ed-d9b98f46a789"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                          review_id                          order_id  \\\n",
              "0  7bc2406110b926393aa56f80a40eba40  73fc7af87114b39712e6da79b0a377eb   \n",
              "1  80e641a11e56f04c1ad469d5645fdfde  a548910a1c6147796b98fdf73dbeba33   \n",
              "2  228ce5500dc1d8e020d8d1322874b6f0  f9e4b658b201a9f2ecdecbb34bed034b   \n",
              "3  e64fb393e7b32834bb789ff8bb30750e  658677c97b385a9be170737859d3511b   \n",
              "4  f7c4243c7fe1938f181bec41a392bdeb  8e6bfb81e283fa7e4f11123a3fb894f1   \n",
              "\n",
              "   review_score review_comment_title  \\\n",
              "0             4     no comment title   \n",
              "1             5     no comment title   \n",
              "2             5     no comment title   \n",
              "3             5     no comment title   \n",
              "4             5     no comment title   \n",
              "\n",
              "                              review_comment_message review_creation_date  \\\n",
              "0                                 no comment message  2018-01-18 00:00:00   \n",
              "1                                 no comment message  2018-03-10 00:00:00   \n",
              "2                                 no comment message  2018-02-17 00:00:00   \n",
              "3              Recebi bem antes do prazo estipulado.  2017-04-21 00:00:00   \n",
              "4  Parabéns lojas lannister adorei comprar pela I...  2018-03-01 00:00:00   \n",
              "\n",
              "  review_answer_timestamp  \n",
              "0     2018-01-18 21:46:59  \n",
              "1     2018-03-11 03:05:13  \n",
              "2     2018-02-18 14:36:24  \n",
              "3     2017-04-21 22:02:06  \n",
              "4     2018-03-02 10:26:53  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-83fbc511-890b-4a26-8260-6921edbbe6bd\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>review_id</th>\n",
              "      <th>order_id</th>\n",
              "      <th>review_score</th>\n",
              "      <th>review_comment_title</th>\n",
              "      <th>review_comment_message</th>\n",
              "      <th>review_creation_date</th>\n",
              "      <th>review_answer_timestamp</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>7bc2406110b926393aa56f80a40eba40</td>\n",
              "      <td>73fc7af87114b39712e6da79b0a377eb</td>\n",
              "      <td>4</td>\n",
              "      <td>no comment title</td>\n",
              "      <td>no comment message</td>\n",
              "      <td>2018-01-18 00:00:00</td>\n",
              "      <td>2018-01-18 21:46:59</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>80e641a11e56f04c1ad469d5645fdfde</td>\n",
              "      <td>a548910a1c6147796b98fdf73dbeba33</td>\n",
              "      <td>5</td>\n",
              "      <td>no comment title</td>\n",
              "      <td>no comment message</td>\n",
              "      <td>2018-03-10 00:00:00</td>\n",
              "      <td>2018-03-11 03:05:13</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>228ce5500dc1d8e020d8d1322874b6f0</td>\n",
              "      <td>f9e4b658b201a9f2ecdecbb34bed034b</td>\n",
              "      <td>5</td>\n",
              "      <td>no comment title</td>\n",
              "      <td>no comment message</td>\n",
              "      <td>2018-02-17 00:00:00</td>\n",
              "      <td>2018-02-18 14:36:24</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>e64fb393e7b32834bb789ff8bb30750e</td>\n",
              "      <td>658677c97b385a9be170737859d3511b</td>\n",
              "      <td>5</td>\n",
              "      <td>no comment title</td>\n",
              "      <td>Recebi bem antes do prazo estipulado.</td>\n",
              "      <td>2017-04-21 00:00:00</td>\n",
              "      <td>2017-04-21 22:02:06</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>f7c4243c7fe1938f181bec41a392bdeb</td>\n",
              "      <td>8e6bfb81e283fa7e4f11123a3fb894f1</td>\n",
              "      <td>5</td>\n",
              "      <td>no comment title</td>\n",
              "      <td>Parabéns lojas lannister adorei comprar pela I...</td>\n",
              "      <td>2018-03-01 00:00:00</td>\n",
              "      <td>2018-03-02 10:26:53</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-83fbc511-890b-4a26-8260-6921edbbe6bd')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-83fbc511-890b-4a26-8260-6921edbbe6bd button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-83fbc511-890b-4a26-8260-6921edbbe6bd');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "reviews.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vg3GQRA06YmR",
        "outputId": "77205c2e-3103-4c93-bf9e-f0fb68e1c5f7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 99224 entries, 0 to 99223\n",
            "Data columns (total 7 columns):\n",
            " #   Column                   Non-Null Count  Dtype \n",
            "---  ------                   --------------  ----- \n",
            " 0   review_id                99224 non-null  object\n",
            " 1   order_id                 99224 non-null  object\n",
            " 2   review_score             99224 non-null  int64 \n",
            " 3   review_comment_title     99224 non-null  object\n",
            " 4   review_comment_message   99224 non-null  object\n",
            " 5   review_creation_date     99224 non-null  object\n",
            " 6   review_answer_timestamp  99224 non-null  object\n",
            "dtypes: int64(1), object(6)\n",
            "memory usage: 6.1+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Transforming Data"
      ],
      "metadata": {
        "id": "xqU4WMY3Uu70"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "times = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',\n",
        "         'order_estimated_delivery_date', 'order_estimated_delivery_date']\n",
        "\n",
        "for col in times:\n",
        "  orders[col]=pd.to_datetime(orders[col])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fh8EDclrkQ2b",
        "outputId": "53b10826-5f89-4ed2-f3d3-d13d4986f89f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  \"\"\"\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  \"\"\"\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "orders.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TyRv2M1nUpn5",
        "outputId": "eec796b4-c359-4dd5-fb3d-e82c80783fca"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 96461 entries, 0 to 99440\n",
            "Data columns (total 8 columns):\n",
            " #   Column                         Non-Null Count  Dtype         \n",
            "---  ------                         --------------  -----         \n",
            " 0   order_id                       96461 non-null  object        \n",
            " 1   customer_id                    96461 non-null  object        \n",
            " 2   order_status                   96461 non-null  object        \n",
            " 3   order_purchase_timestamp       96461 non-null  datetime64[ns]\n",
            " 4   order_approved_at              96461 non-null  datetime64[ns]\n",
            " 5   order_delivered_carrier_date   96461 non-null  datetime64[ns]\n",
            " 6   order_delivered_customer_date  96461 non-null  datetime64[ns]\n",
            " 7   order_estimated_delivery_date  96461 non-null  datetime64[ns]\n",
            "dtypes: datetime64[ns](5), object(3)\n",
            "memory usage: 6.6+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Merging Data"
      ],
      "metadata": {
        "id": "EKY3OX97WzR9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "order = pd.merge(customers,orders[['order_id','customer_id','order_purchase_timestamp','order_approved_at']],on='customer_id')\n",
        "order.head()"
      ],
      "metadata": {
        "id": "-Bg7Zc1JZhuE",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 305
        },
        "outputId": "388492d2-c4de-4a5d-dc01-bee7e1a51906"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                        customer_id                customer_unique_id  \\\n",
              "0  06b8999e2fba1a1fbc88172c00ba8bc7  861eff4711a542e4b93843c6dd7febb0   \n",
              "1  18955e83d337fd6b2def6b18a428ac77  290c77bc529b7ac935b93aa66c333dc3   \n",
              "2  4e7b3e00288586ebd08712fdd0374a03  060e732b5b29e8181a18229c7b0b2b5e   \n",
              "3  b2b6027bc5c5109e529d4dc6358b12c3  259dac757896d24d7702b9acbbff3f3c   \n",
              "4  4f2d8ab171c80ec8364f7c12e35b23ad  345ecd01c38d18a9036ed96c73b8d066   \n",
              "\n",
              "   customer_zip_code_prefix          customer_city customer_state  \\\n",
              "0                     14409                 franca             SP   \n",
              "1                      9790  sao bernardo do campo             SP   \n",
              "2                      1151              sao paulo             SP   \n",
              "3                      8775        mogi das cruzes             SP   \n",
              "4                     13056               campinas             SP   \n",
              "\n",
              "                           order_id order_purchase_timestamp  \\\n",
              "0  00e7ee1b050b8499577073aeb2a297a1      2017-05-16 15:05:35   \n",
              "1  29150127e6685892b6eab3eec79f59c7      2018-01-12 20:48:24   \n",
              "2  b2059ed67ce144a36e2aa97d2c9e9ad2      2018-05-19 16:07:45   \n",
              "3  951670f92359f4fe4a63112aa7306eba      2018-03-13 16:06:38   \n",
              "4  6b7d50bd145f6fc7f33cebabd7e49d0f      2018-07-29 09:51:30   \n",
              "\n",
              "    order_approved_at  \n",
              "0 2017-05-16 15:22:12  \n",
              "1 2018-01-12 20:58:32  \n",
              "2 2018-05-20 16:19:10  \n",
              "3 2018-03-13 17:29:19  \n",
              "4 2018-07-29 10:10:09  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d13c5767-62cd-4e0f-84f7-27a6c84426ad\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>customer_id</th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>customer_zip_code_prefix</th>\n",
              "      <th>customer_city</th>\n",
              "      <th>customer_state</th>\n",
              "      <th>order_id</th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "      <th>order_approved_at</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>\n",
              "      <td>861eff4711a542e4b93843c6dd7febb0</td>\n",
              "      <td>14409</td>\n",
              "      <td>franca</td>\n",
              "      <td>SP</td>\n",
              "      <td>00e7ee1b050b8499577073aeb2a297a1</td>\n",
              "      <td>2017-05-16 15:05:35</td>\n",
              "      <td>2017-05-16 15:22:12</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18955e83d337fd6b2def6b18a428ac77</td>\n",
              "      <td>290c77bc529b7ac935b93aa66c333dc3</td>\n",
              "      <td>9790</td>\n",
              "      <td>sao bernardo do campo</td>\n",
              "      <td>SP</td>\n",
              "      <td>29150127e6685892b6eab3eec79f59c7</td>\n",
              "      <td>2018-01-12 20:48:24</td>\n",
              "      <td>2018-01-12 20:58:32</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4e7b3e00288586ebd08712fdd0374a03</td>\n",
              "      <td>060e732b5b29e8181a18229c7b0b2b5e</td>\n",
              "      <td>1151</td>\n",
              "      <td>sao paulo</td>\n",
              "      <td>SP</td>\n",
              "      <td>b2059ed67ce144a36e2aa97d2c9e9ad2</td>\n",
              "      <td>2018-05-19 16:07:45</td>\n",
              "      <td>2018-05-20 16:19:10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>b2b6027bc5c5109e529d4dc6358b12c3</td>\n",
              "      <td>259dac757896d24d7702b9acbbff3f3c</td>\n",
              "      <td>8775</td>\n",
              "      <td>mogi das cruzes</td>\n",
              "      <td>SP</td>\n",
              "      <td>951670f92359f4fe4a63112aa7306eba</td>\n",
              "      <td>2018-03-13 16:06:38</td>\n",
              "      <td>2018-03-13 17:29:19</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4f2d8ab171c80ec8364f7c12e35b23ad</td>\n",
              "      <td>345ecd01c38d18a9036ed96c73b8d066</td>\n",
              "      <td>13056</td>\n",
              "      <td>campinas</td>\n",
              "      <td>SP</td>\n",
              "      <td>6b7d50bd145f6fc7f33cebabd7e49d0f</td>\n",
              "      <td>2018-07-29 09:51:30</td>\n",
              "      <td>2018-07-29 10:10:09</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d13c5767-62cd-4e0f-84f7-27a6c84426ad')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d13c5767-62cd-4e0f-84f7-27a6c84426ad button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d13c5767-62cd-4e0f-84f7-27a6c84426ad');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_temp = orders.merge(customers, on = 'customer_id')\n",
        "df_temp = df_temp.merge(payments, on = 'order_id')\n",
        "df_temp = df_temp.merge(orders, on = 'order_id')\n",
        "df_temp.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 299
        },
        "id": "XMr9lxQ4vK-b",
        "outputId": "782fe61f-613c-4688-d689-d2bc8812c26b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                           order_id                     customer_id_x  \\\n",
              "0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "1  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "2  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "3  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef   \n",
              "4  47770eb9100c2d0c44946d9cf07ec65d  41ce2a54c0b03bf3443c3d931a367089   \n",
              "\n",
              "  order_status_x order_purchase_timestamp_x order_approved_at_x  \\\n",
              "0      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "1      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "2      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "3      delivered        2018-07-24 20:41:37 2018-07-26 03:24:27   \n",
              "4      delivered        2018-08-08 08:38:49 2018-08-08 08:55:23   \n",
              "\n",
              "  order_delivered_carrier_date_x order_delivered_customer_date_x  \\\n",
              "0            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "1            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "2            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "3            2018-07-26 14:31:00             2018-08-07 15:27:45   \n",
              "4            2018-08-08 13:50:00             2018-08-17 18:06:29   \n",
              "\n",
              "  order_estimated_delivery_date_x                customer_unique_id  \\\n",
              "0                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "1                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "2                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "3                      2018-08-13  af07308b275d755c9edb36a90c618231   \n",
              "4                      2018-09-04  3a653a41f6f9fc3d2a113cf8398680e8   \n",
              "\n",
              "   customer_zip_code_prefix  ... payment_type payment_installments  \\\n",
              "0                      3149  ...  credit_card                    1   \n",
              "1                      3149  ...      voucher                    1   \n",
              "2                      3149  ...      voucher                    1   \n",
              "3                     47813  ...       boleto                    1   \n",
              "4                     75265  ...  credit_card                    3   \n",
              "\n",
              "   payment_value                     customer_id_y  order_status_y  \\\n",
              "0          18.12  9ef432eb6251297304e76186b10a928d       delivered   \n",
              "1           2.00  9ef432eb6251297304e76186b10a928d       delivered   \n",
              "2          18.59  9ef432eb6251297304e76186b10a928d       delivered   \n",
              "3         141.46  b0830fb4747a6c6d20dea0b8c802d7ef       delivered   \n",
              "4         179.12  41ce2a54c0b03bf3443c3d931a367089       delivered   \n",
              "\n",
              "   order_purchase_timestamp_y order_approved_at_y  \\\n",
              "0         2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "1         2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "2         2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "3         2018-07-24 20:41:37 2018-07-26 03:24:27   \n",
              "4         2018-08-08 08:38:49 2018-08-08 08:55:23   \n",
              "\n",
              "  order_delivered_carrier_date_y order_delivered_customer_date_y  \\\n",
              "0            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "1            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "2            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "3            2018-07-26 14:31:00             2018-08-07 15:27:45   \n",
              "4            2018-08-08 13:50:00             2018-08-17 18:06:29   \n",
              "\n",
              "  order_estimated_delivery_date_y  \n",
              "0                      2017-10-18  \n",
              "1                      2017-10-18  \n",
              "2                      2017-10-18  \n",
              "3                      2018-08-13  \n",
              "4                      2018-09-04  \n",
              "\n",
              "[5 rows x 23 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b146a4af-1c18-4cce-81e0-00a80b166373\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_id</th>\n",
              "      <th>customer_id_x</th>\n",
              "      <th>order_status_x</th>\n",
              "      <th>order_purchase_timestamp_x</th>\n",
              "      <th>order_approved_at_x</th>\n",
              "      <th>order_delivered_carrier_date_x</th>\n",
              "      <th>order_delivered_customer_date_x</th>\n",
              "      <th>order_estimated_delivery_date_x</th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>customer_zip_code_prefix</th>\n",
              "      <th>...</th>\n",
              "      <th>payment_type</th>\n",
              "      <th>payment_installments</th>\n",
              "      <th>payment_value</th>\n",
              "      <th>customer_id_y</th>\n",
              "      <th>order_status_y</th>\n",
              "      <th>order_purchase_timestamp_y</th>\n",
              "      <th>order_approved_at_y</th>\n",
              "      <th>order_delivered_carrier_date_y</th>\n",
              "      <th>order_delivered_customer_date_y</th>\n",
              "      <th>order_estimated_delivery_date_y</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>credit_card</td>\n",
              "      <td>1</td>\n",
              "      <td>18.12</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>voucher</td>\n",
              "      <td>1</td>\n",
              "      <td>2.00</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>voucher</td>\n",
              "      <td>1</td>\n",
              "      <td>18.59</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>53cdb2fc8bc7dce0b6741e2150273451</td>\n",
              "      <td>b0830fb4747a6c6d20dea0b8c802d7ef</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "      <td>af07308b275d755c9edb36a90c618231</td>\n",
              "      <td>47813</td>\n",
              "      <td>...</td>\n",
              "      <td>boleto</td>\n",
              "      <td>1</td>\n",
              "      <td>141.46</td>\n",
              "      <td>b0830fb4747a6c6d20dea0b8c802d7ef</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>47770eb9100c2d0c44946d9cf07ec65d</td>\n",
              "      <td>41ce2a54c0b03bf3443c3d931a367089</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-08-08 08:38:49</td>\n",
              "      <td>2018-08-08 08:55:23</td>\n",
              "      <td>2018-08-08 13:50:00</td>\n",
              "      <td>2018-08-17 18:06:29</td>\n",
              "      <td>2018-09-04</td>\n",
              "      <td>3a653a41f6f9fc3d2a113cf8398680e8</td>\n",
              "      <td>75265</td>\n",
              "      <td>...</td>\n",
              "      <td>credit_card</td>\n",
              "      <td>3</td>\n",
              "      <td>179.12</td>\n",
              "      <td>41ce2a54c0b03bf3443c3d931a367089</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-08-08 08:38:49</td>\n",
              "      <td>2018-08-08 08:55:23</td>\n",
              "      <td>2018-08-08 13:50:00</td>\n",
              "      <td>2018-08-17 18:06:29</td>\n",
              "      <td>2018-09-04</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 23 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b146a4af-1c18-4cce-81e0-00a80b166373')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b146a4af-1c18-4cce-81e0-00a80b166373 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b146a4af-1c18-4cce-81e0-00a80b166373');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "payment = payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()\n",
        "payment.head()"
      ],
      "metadata": {
        "id": "jugaT3zsbHWB",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "3651b4f4-1f3a-4741-843d-c2ef70115fae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                           order_id  payment_value\n",
              "0  00010242fe8c5a6d1ba2dd792cb16214          72.19\n",
              "1  00018f77f2f0320c557190d7a144bdd3         259.83\n",
              "2  000229ec398224ef6ca0657da4fc703e         216.87\n",
              "3  00024acbcdf0a6daa1e931b038114c75          25.78\n",
              "4  00042b26cf59d7ce69dfabb4e55b4fd9         218.04"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-14f25391-a861-4350-b0ce-555febd89126\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_id</th>\n",
              "      <th>payment_value</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>00010242fe8c5a6d1ba2dd792cb16214</td>\n",
              "      <td>72.19</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>00018f77f2f0320c557190d7a144bdd3</td>\n",
              "      <td>259.83</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>000229ec398224ef6ca0657da4fc703e</td>\n",
              "      <td>216.87</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>00024acbcdf0a6daa1e931b038114c75</td>\n",
              "      <td>25.78</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>00042b26cf59d7ce69dfabb4e55b4fd9</td>\n",
              "      <td>218.04</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-14f25391-a861-4350-b0ce-555febd89126')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-14f25391-a861-4350-b0ce-555febd89126 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-14f25391-a861-4350-b0ce-555febd89126');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_temp.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ff-OHs7tMkx",
        "outputId": "8af269c4-68a4-4cdd-9cf6-c89d66fa1a56"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 100739 entries, 0 to 100738\n",
            "Data columns (total 23 columns):\n",
            " #   Column                           Non-Null Count   Dtype         \n",
            "---  ------                           --------------   -----         \n",
            " 0   order_id                         100739 non-null  object        \n",
            " 1   customer_id_x                    100739 non-null  object        \n",
            " 2   order_status_x                   100739 non-null  object        \n",
            " 3   order_purchase_timestamp_x       100739 non-null  datetime64[ns]\n",
            " 4   order_approved_at_x              100739 non-null  datetime64[ns]\n",
            " 5   order_delivered_carrier_date_x   100739 non-null  datetime64[ns]\n",
            " 6   order_delivered_customer_date_x  100739 non-null  datetime64[ns]\n",
            " 7   order_estimated_delivery_date_x  100739 non-null  datetime64[ns]\n",
            " 8   customer_unique_id               100739 non-null  object        \n",
            " 9   customer_zip_code_prefix         100739 non-null  int64         \n",
            " 10  customer_city                    100739 non-null  object        \n",
            " 11  customer_state                   100739 non-null  object        \n",
            " 12  payment_sequential               100739 non-null  int64         \n",
            " 13  payment_type                     100739 non-null  object        \n",
            " 14  payment_installments             100739 non-null  int64         \n",
            " 15  payment_value                    100739 non-null  float64       \n",
            " 16  customer_id_y                    100739 non-null  object        \n",
            " 17  order_status_y                   100739 non-null  object        \n",
            " 18  order_purchase_timestamp_y       100739 non-null  datetime64[ns]\n",
            " 19  order_approved_at_y              100739 non-null  datetime64[ns]\n",
            " 20  order_delivered_carrier_date_y   100739 non-null  datetime64[ns]\n",
            " 21  order_delivered_customer_date_y  100739 non-null  datetime64[ns]\n",
            " 22  order_estimated_delivery_date_y  100739 non-null  datetime64[ns]\n",
            "dtypes: datetime64[ns](10), float64(1), int64(3), object(9)\n",
            "memory usage: 18.4+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#marge data sellers dengan orderid\n",
        "seller_seg = pd.merge(items[['order_id','seller_id','price']],orders[['order_id','order_purchase_timestamp','order_delivered_carrier_date']],on='order_id')\n",
        "\n",
        "#marge data review score dengan order dan order id\n",
        "seller_seg = pd.merge(seller_seg,reviews[['order_id','review_score']],on='order_id')"
      ],
      "metadata": {
        "id": "jvJeCbqARBlu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#mengubah format dates dengan datetime\n",
        "seller_seg['order_purchase_timestamp'] = pd.to_datetime(seller_seg['order_purchase_timestamp']).dt.date\n",
        "seller_seg['order_delivered_carrier_date'] = pd.to_datetime(seller_seg['order_delivered_carrier_date']).dt.date\n",
        "seller_seg['days_to_del'] = (seller_seg['order_delivered_carrier_date']-seller_seg['order_purchase_timestamp']).dt.days"
      ],
      "metadata": {
        "id": "XPkKFTvPS6YR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#menghapus kolom yang tidak digunakkan\n",
        "seller_seg.drop(['order_purchase_timestamp','order_delivered_carrier_date'],axis=1,inplace=True)\n",
        "\n",
        "#menghapus tanggal yang tidak sesuai\n",
        "seller_seg_1 = seller_seg[seller_seg['days_to_del']>=0]"
      ],
      "metadata": {
        "id": "-J2ALTFuSncu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "seller_seg_2 = seller_seg_1.groupby('seller_id').agg({'order_id':'count','price':'sum','review_score':'mean','days_to_del':'mean'})\n",
        "seller_seg_2.columns = ['Tot_sales','Tot_amount','Avg_review','Avg_delivery']"
      ],
      "metadata": {
        "id": "4OEjbsA7Tp8Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#mengisi nilai missing value dengan nilai rata-rata\n",
        "seller_seg_2['Avg_delivery'].fillna(seller_seg_2['Avg_delivery'].mean(),inplace=True)\n",
        "quantiles = seller_seg_2.quantile(q=[0.25,0.5,0.75])\n",
        "quantiles.to_dict()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x3MWkeaNUK6H",
        "outputId": "8fd79e2a-9e93-41a1-bf56-d850a9c38d3f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Tot_sales': {0.25: 2.0, 0.5: 8.0, 0.75: 25.0},\n",
              " 'Tot_amount': {0.25: 219.9, 0.5: 843.8, 0.75: 3474.2999999999997},\n",
              " 'Avg_review': {0.25: 3.891304347826087, 0.5: 4.2727272727272725, 0.75: 4.7},\n",
              " 'Avg_delivery': {0.25: 1.7894736842105263,\n",
              "  0.5: 2.6330434782608694,\n",
              "  0.75: 4.0}}"
            ]
          },
          "metadata": {},
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)\n",
        "def RScore(x,p,d):\n",
        "    if x <= d[p][0.25]:\n",
        "        return 4\n",
        "    elif x <= d[p][0.50]:\n",
        "        return 3\n",
        "    elif x <= d[p][0.75]: \n",
        "        return 2\n",
        "    else:\n",
        "        return 1\n",
        "# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)\n",
        "def FMScore(x,p,d):\n",
        "    if x <= d[p][0.25]:\n",
        "        return 1\n",
        "    elif x <= d[p][0.50]:\n",
        "        return 2\n",
        "    elif x <= d[p][0.75]: \n",
        "        return 3\n",
        "    else:\n",
        "        return 4"
      ],
      "metadata": {
        "id": "V3MlEbCqoA6Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "seller_seg_2['Tot_sales_Quartile'] = seller_seg_2['Tot_sales'].apply(FMScore, args=('Tot_sales',quantiles,))\n",
        "seller_seg_2['Tot_amount_Quartile'] = seller_seg_2['Tot_amount'].apply(FMScore, args=('Tot_amount',quantiles,))\n",
        "seller_seg_2['Avg_review_Quartile'] = seller_seg_2['Avg_review'].apply(FMScore, args=('Avg_review',quantiles,))\n",
        "seller_seg_2['Avg_delivery_Quartile'] = seller_seg_2['Avg_delivery'].apply(RScore, args=('Avg_delivery',quantiles,))"
      ],
      "metadata": {
        "id": "8i4ycwzmUZ-e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "seller_seg_2['SellerScore'] = seller_seg_2['Tot_sales_Quartile'].map(str) \\\n",
        "                            + seller_seg_2['Tot_amount_Quartile'].map(str) \\\n",
        "                            + seller_seg_2['Avg_review_Quartile'] .map(str) \\\n",
        "                            + seller_seg_2['Avg_delivery_Quartile'].map(str)\n",
        "seller_seg_2.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "id": "IzVxuwpzUgTu",
        "outputId": "f7b69032-3ce9-4eeb-ecf1-ddf7f3936100"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                  Tot_sales  Tot_amount  Avg_review  \\\n",
              "seller_id                                                             \n",
              "0015a82c2db000af6aaaf3ae2ecb0532          3     2685.00    3.666667   \n",
              "001cca7ae9ae17fb1caed9dfb1094831        231    24177.03    3.965368   \n",
              "002100f778ceb8431b7a1020ff7ab48f         55     1236.50    4.036364   \n",
              "003554e2dce176b5555353e4f3555ac8          1      120.00    5.000000   \n",
              "004c9cd9d87a3c30c522c48c4fc07416        171    20017.23    4.152047   \n",
              "\n",
              "                                  Avg_delivery  Tot_sales_Quartile  \\\n",
              "seller_id                                                            \n",
              "0015a82c2db000af6aaaf3ae2ecb0532      3.333333                   2   \n",
              "001cca7ae9ae17fb1caed9dfb1094831      2.627706                   4   \n",
              "002100f778ceb8431b7a1020ff7ab48f      4.345455                   4   \n",
              "003554e2dce176b5555353e4f3555ac8      0.000000                   1   \n",
              "004c9cd9d87a3c30c522c48c4fc07416      1.649123                   4   \n",
              "\n",
              "                                  Tot_amount_Quartile  Avg_review_Quartile  \\\n",
              "seller_id                                                                    \n",
              "0015a82c2db000af6aaaf3ae2ecb0532                    3                    1   \n",
              "001cca7ae9ae17fb1caed9dfb1094831                    4                    2   \n",
              "002100f778ceb8431b7a1020ff7ab48f                    3                    2   \n",
              "003554e2dce176b5555353e4f3555ac8                    1                    4   \n",
              "004c9cd9d87a3c30c522c48c4fc07416                    4                    2   \n",
              "\n",
              "                                  Avg_delivery_Quartile SellerScore  \n",
              "seller_id                                                            \n",
              "0015a82c2db000af6aaaf3ae2ecb0532                      2        2312  \n",
              "001cca7ae9ae17fb1caed9dfb1094831                      3        4423  \n",
              "002100f778ceb8431b7a1020ff7ab48f                      1        4321  \n",
              "003554e2dce176b5555353e4f3555ac8                      4        1144  \n",
              "004c9cd9d87a3c30c522c48c4fc07416                      4        4424  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-30ded9a9-bd5c-4e92-8c76-3e7fdd9c849f\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Tot_sales</th>\n",
              "      <th>Tot_amount</th>\n",
              "      <th>Avg_review</th>\n",
              "      <th>Avg_delivery</th>\n",
              "      <th>Tot_sales_Quartile</th>\n",
              "      <th>Tot_amount_Quartile</th>\n",
              "      <th>Avg_review_Quartile</th>\n",
              "      <th>Avg_delivery_Quartile</th>\n",
              "      <th>SellerScore</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>seller_id</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0015a82c2db000af6aaaf3ae2ecb0532</th>\n",
              "      <td>3</td>\n",
              "      <td>2685.00</td>\n",
              "      <td>3.666667</td>\n",
              "      <td>3.333333</td>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>2312</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>001cca7ae9ae17fb1caed9dfb1094831</th>\n",
              "      <td>231</td>\n",
              "      <td>24177.03</td>\n",
              "      <td>3.965368</td>\n",
              "      <td>2.627706</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>4423</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>002100f778ceb8431b7a1020ff7ab48f</th>\n",
              "      <td>55</td>\n",
              "      <td>1236.50</td>\n",
              "      <td>4.036364</td>\n",
              "      <td>4.345455</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>4321</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>003554e2dce176b5555353e4f3555ac8</th>\n",
              "      <td>1</td>\n",
              "      <td>120.00</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>1144</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>004c9cd9d87a3c30c522c48c4fc07416</th>\n",
              "      <td>171</td>\n",
              "      <td>20017.23</td>\n",
              "      <td>4.152047</td>\n",
              "      <td>1.649123</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>4</td>\n",
              "      <td>4424</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-30ded9a9-bd5c-4e92-8c76-3e7fdd9c849f')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-30ded9a9-bd5c-4e92-8c76-3e7fdd9c849f button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-30ded9a9-bd5c-4e92-8c76-3e7fdd9c849f');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#mencari seller berdasarkan Best Sellers, highest Reviewed Sellers, Fastest delivering sellers\n",
        "print(\"Best Sellers: \",len(seller_seg_2[seller_seg_2['SellerScore']=='4444']))\n",
        "print('highest Reviewed Sellers: ',len(seller_seg_2[seller_seg_2['Avg_review_Quartile']==4]))\n",
        "print(\"Fastest delivering sellers: \",len(seller_seg_2[seller_seg_2['Avg_delivery_Quartile']==4]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wySU-H3vUoyF",
        "outputId": "96b54682-5fa2-4196-88f8-402d9402452b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best Sellers:  5\n",
            "highest Reviewed Sellers:  736\n",
            "Fastest delivering sellers:  742\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fastest_delivering = seller_seg_2[seller_seg_2['Avg_delivery_Quartile']==4].reset_index()\n",
        "highest_reviewed = seller_seg_2[seller_seg_2['Avg_review_Quartile']==4].reset_index()\n",
        "len(set(list(fastest_delivering['seller_id'])).intersection(highest_reviewed['seller_id'].tolist()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZymRH3cpUzon",
        "outputId": "d992f2b6-c809-47dc-cc06-4cc84093f4cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "245"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm = pd.merge(order,payment,on='order_id')\n",
        "rfm.drop(['customer_zip_code_prefix','customer_city','customer_state'],axis=1,inplace=True)\n",
        "rfm['order_purchase_timestamp']=pd.to_datetime(rfm['order_purchase_timestamp']).dt.date\n",
        "\n",
        "rfm.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 270
        },
        "id": "I73amEHl_kKJ",
        "outputId": "7bf9e412-a3fe-4cc8-ef9a-524ad8c35a47"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                        customer_id                customer_unique_id  \\\n",
              "0  06b8999e2fba1a1fbc88172c00ba8bc7  861eff4711a542e4b93843c6dd7febb0   \n",
              "1  18955e83d337fd6b2def6b18a428ac77  290c77bc529b7ac935b93aa66c333dc3   \n",
              "2  4e7b3e00288586ebd08712fdd0374a03  060e732b5b29e8181a18229c7b0b2b5e   \n",
              "3  b2b6027bc5c5109e529d4dc6358b12c3  259dac757896d24d7702b9acbbff3f3c   \n",
              "4  4f2d8ab171c80ec8364f7c12e35b23ad  345ecd01c38d18a9036ed96c73b8d066   \n",
              "\n",
              "                           order_id order_purchase_timestamp  \\\n",
              "0  00e7ee1b050b8499577073aeb2a297a1               2017-05-16   \n",
              "1  29150127e6685892b6eab3eec79f59c7               2018-01-12   \n",
              "2  b2059ed67ce144a36e2aa97d2c9e9ad2               2018-05-19   \n",
              "3  951670f92359f4fe4a63112aa7306eba               2018-03-13   \n",
              "4  6b7d50bd145f6fc7f33cebabd7e49d0f               2018-07-29   \n",
              "\n",
              "    order_approved_at  payment_value  \n",
              "0 2017-05-16 15:22:12         146.87  \n",
              "1 2018-01-12 20:58:32         335.48  \n",
              "2 2018-05-20 16:19:10         157.73  \n",
              "3 2018-03-13 17:29:19         173.30  \n",
              "4 2018-07-29 10:10:09         252.25  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-02f6e9ee-14fe-4a94-9462-eea5068f6307\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>customer_id</th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>order_id</th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "      <th>order_approved_at</th>\n",
              "      <th>payment_value</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>\n",
              "      <td>861eff4711a542e4b93843c6dd7febb0</td>\n",
              "      <td>00e7ee1b050b8499577073aeb2a297a1</td>\n",
              "      <td>2017-05-16</td>\n",
              "      <td>2017-05-16 15:22:12</td>\n",
              "      <td>146.87</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18955e83d337fd6b2def6b18a428ac77</td>\n",
              "      <td>290c77bc529b7ac935b93aa66c333dc3</td>\n",
              "      <td>29150127e6685892b6eab3eec79f59c7</td>\n",
              "      <td>2018-01-12</td>\n",
              "      <td>2018-01-12 20:58:32</td>\n",
              "      <td>335.48</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4e7b3e00288586ebd08712fdd0374a03</td>\n",
              "      <td>060e732b5b29e8181a18229c7b0b2b5e</td>\n",
              "      <td>b2059ed67ce144a36e2aa97d2c9e9ad2</td>\n",
              "      <td>2018-05-19</td>\n",
              "      <td>2018-05-20 16:19:10</td>\n",
              "      <td>157.73</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>b2b6027bc5c5109e529d4dc6358b12c3</td>\n",
              "      <td>259dac757896d24d7702b9acbbff3f3c</td>\n",
              "      <td>951670f92359f4fe4a63112aa7306eba</td>\n",
              "      <td>2018-03-13</td>\n",
              "      <td>2018-03-13 17:29:19</td>\n",
              "      <td>173.30</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4f2d8ab171c80ec8364f7c12e35b23ad</td>\n",
              "      <td>345ecd01c38d18a9036ed96c73b8d066</td>\n",
              "      <td>6b7d50bd145f6fc7f33cebabd7e49d0f</td>\n",
              "      <td>2018-07-29</td>\n",
              "      <td>2018-07-29 10:10:09</td>\n",
              "      <td>252.25</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-02f6e9ee-14fe-4a94-9462-eea5068f6307')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-02f6e9ee-14fe-4a94-9462-eea5068f6307 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-02f6e9ee-14fe-4a94-9462-eea5068f6307');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JR6Gft6gzpk7",
        "outputId": "6244674d-f445-4c4d-c6e8-03b692a0e99b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 96460 entries, 0 to 96459\n",
            "Data columns (total 6 columns):\n",
            " #   Column                    Non-Null Count  Dtype         \n",
            "---  ------                    --------------  -----         \n",
            " 0   customer_id               96460 non-null  object        \n",
            " 1   customer_unique_id        96460 non-null  object        \n",
            " 2   order_id                  96460 non-null  object        \n",
            " 3   order_purchase_timestamp  96460 non-null  object        \n",
            " 4   order_approved_at         96460 non-null  datetime64[ns]\n",
            " 5   payment_value             96460 non-null  float64       \n",
            "dtypes: datetime64[ns](1), float64(1), object(4)\n",
            "memory usage: 5.2+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Engineering"
      ],
      "metadata": {
        "id": "3YNoEEwC72MF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Recency"
      ],
      "metadata": {
        "id": "IXPmpVFZ1HJk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "recency = pd.DataFrame(rfm.groupby('customer_unique_id')['order_purchase_timestamp'].max())\n",
        "recency.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "id": "nze2sBS11JG5",
        "outputId": "e1918b38-3990-49f6-f5b6-8c7c060ee708"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                 order_purchase_timestamp\n",
              "customer_unique_id                                       \n",
              "0000366f3b9a7992bf8c76cfdf3221e2               2018-05-10\n",
              "0000b849f77a49e4a4ce2b2a4ca5be3f               2018-05-07\n",
              "0000f46a3911fa3c0805444483337064               2017-03-10\n",
              "0000f6ccb0745a6a4b88665a16c9f078               2017-10-12\n",
              "0004aac84e0df4da2b147fca70cf8255               2017-11-14"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ba73cb83-38c8-40b5-b8bc-86f9b3edc287\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0000366f3b9a7992bf8c76cfdf3221e2</th>\n",
              "      <td>2018-05-10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000b849f77a49e4a4ce2b2a4ca5be3f</th>\n",
              "      <td>2018-05-07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f46a3911fa3c0805444483337064</th>\n",
              "      <td>2017-03-10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f6ccb0745a6a4b88665a16c9f078</th>\n",
              "      <td>2017-10-12</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0004aac84e0df4da2b147fca70cf8255</th>\n",
              "      <td>2017-11-14</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ba73cb83-38c8-40b5-b8bc-86f9b3edc287')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ba73cb83-38c8-40b5-b8bc-86f9b3edc287 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ba73cb83-38c8-40b5-b8bc-86f9b3edc287');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "recency['recent_days'] = recency['order_purchase_timestamp'].max()-recency['order_purchase_timestamp']\n",
        "recency['recent_days'] = recency['recent_days'].dt.days\n",
        "recency.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "id": "0Jm_17YH1Rdj",
        "outputId": "ccbabddc-6999-4a9c-f7aa-66ed45c4e8e5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                 order_purchase_timestamp  recent_days\n",
              "customer_unique_id                                                    \n",
              "0000366f3b9a7992bf8c76cfdf3221e2               2018-05-10          111\n",
              "0000b849f77a49e4a4ce2b2a4ca5be3f               2018-05-07          114\n",
              "0000f46a3911fa3c0805444483337064               2017-03-10          537\n",
              "0000f6ccb0745a6a4b88665a16c9f078               2017-10-12          321\n",
              "0004aac84e0df4da2b147fca70cf8255               2017-11-14          288"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5075a82a-52ca-4d36-85e0-7693e887be29\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "      <th>recent_days</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0000366f3b9a7992bf8c76cfdf3221e2</th>\n",
              "      <td>2018-05-10</td>\n",
              "      <td>111</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000b849f77a49e4a4ce2b2a4ca5be3f</th>\n",
              "      <td>2018-05-07</td>\n",
              "      <td>114</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f46a3911fa3c0805444483337064</th>\n",
              "      <td>2017-03-10</td>\n",
              "      <td>537</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f6ccb0745a6a4b88665a16c9f078</th>\n",
              "      <td>2017-10-12</td>\n",
              "      <td>321</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0004aac84e0df4da2b147fca70cf8255</th>\n",
              "      <td>2017-11-14</td>\n",
              "      <td>288</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5075a82a-52ca-4d36-85e0-7693e887be29')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-5075a82a-52ca-4d36-85e0-7693e887be29 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5075a82a-52ca-4d36-85e0-7693e887be29');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Frequency"
      ],
      "metadata": {
        "id": "0ITv4PaW1g_C"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "frequency = pd.DataFrame(rfm.groupby('customer_unique_id')['order_approved_at'].count())\n",
        "frequency.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "id": "fnAj6vtB1jDJ",
        "outputId": "8844db23-6850-4824-fca4-b460a07ffaf4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                  order_approved_at\n",
              "customer_unique_id                                 \n",
              "0000366f3b9a7992bf8c76cfdf3221e2                  1\n",
              "0000b849f77a49e4a4ce2b2a4ca5be3f                  1\n",
              "0000f46a3911fa3c0805444483337064                  1\n",
              "0000f6ccb0745a6a4b88665a16c9f078                  1\n",
              "0004aac84e0df4da2b147fca70cf8255                  1"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-07fea2b2-6a44-4219-9c22-c6fcaf5dfc20\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_approved_at</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0000366f3b9a7992bf8c76cfdf3221e2</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000b849f77a49e4a4ce2b2a4ca5be3f</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f46a3911fa3c0805444483337064</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f6ccb0745a6a4b88665a16c9f078</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0004aac84e0df4da2b147fca70cf8255</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-07fea2b2-6a44-4219-9c22-c6fcaf5dfc20')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-07fea2b2-6a44-4219-9c22-c6fcaf5dfc20 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-07fea2b2-6a44-4219-9c22-c6fcaf5dfc20');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Monetary"
      ],
      "metadata": {
        "id": "dfDBw1M31oVn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "monetary = pd.DataFrame(rfm[['customer_unique_id','payment_value']].groupby('customer_unique_id')['payment_value'].sum())\n",
        "monetary.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "id": "vMe6gZTD1qJr",
        "outputId": "5e10d2b5-a933-4d11-cb67-614f91bbbe36"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                  payment_value\n",
              "customer_unique_id                             \n",
              "0000366f3b9a7992bf8c76cfdf3221e2         141.90\n",
              "0000b849f77a49e4a4ce2b2a4ca5be3f          27.19\n",
              "0000f46a3911fa3c0805444483337064          86.22\n",
              "0000f6ccb0745a6a4b88665a16c9f078          43.62\n",
              "0004aac84e0df4da2b147fca70cf8255         196.89"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-226d654d-c813-4267-8758-35527f5ab8fc\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>payment_value</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0000366f3b9a7992bf8c76cfdf3221e2</th>\n",
              "      <td>141.90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000b849f77a49e4a4ce2b2a4ca5be3f</th>\n",
              "      <td>27.19</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f46a3911fa3c0805444483337064</th>\n",
              "      <td>86.22</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0000f6ccb0745a6a4b88665a16c9f078</th>\n",
              "      <td>43.62</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0004aac84e0df4da2b147fca70cf8255</th>\n",
              "      <td>196.89</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-226d654d-c813-4267-8758-35527f5ab8fc')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-226d654d-c813-4267-8758-35527f5ab8fc button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-226d654d-c813-4267-8758-35527f5ab8fc');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 53
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## RFM"
      ],
      "metadata": {
        "id": "TISG55S12H2E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "rfm = pd.merge(recency,frequency,on='customer_unique_id')\n",
        "rfm = pd.merge(rfm,monetary,on='customer_unique_id')"
      ],
      "metadata": {
        "id": "zUXTK4MV2KVL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rfm.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R4J1GwoEqIE8",
        "outputId": "589a0732-f660-457d-c1ce-c597036e007c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Index: 93341 entries, 0000366f3b9a7992bf8c76cfdf3221e2 to ffffd2657e2aad2907e67c3e9daecbeb\n",
            "Data columns (total 4 columns):\n",
            " #   Column                    Non-Null Count  Dtype  \n",
            "---  ------                    --------------  -----  \n",
            " 0   order_purchase_timestamp  93341 non-null  object \n",
            " 1   recent_days               93341 non-null  int64  \n",
            " 2   order_approved_at         93341 non-null  int64  \n",
            " 3   payment_value             93341 non-null  float64\n",
            "dtypes: float64(1), int64(2), object(1)\n",
            "memory usage: 3.6+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm.drop(['order_purchase_timestamp'],axis=1,inplace=True)\n",
        "rfm.reset_index(inplace=True)\n",
        "rfm.columns=['Cust_unique_Id','Recency','Frequency','Monetary']\n",
        "rfm"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "2xbsaJrI2aAw",
        "outputId": "98717b27-be06-42a8-bd54-6c1be65b312e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                         Cust_unique_Id  Recency  Frequency  Monetary\n",
              "0      0000366f3b9a7992bf8c76cfdf3221e2      111          1    141.90\n",
              "1      0000b849f77a49e4a4ce2b2a4ca5be3f      114          1     27.19\n",
              "2      0000f46a3911fa3c0805444483337064      537          1     86.22\n",
              "3      0000f6ccb0745a6a4b88665a16c9f078      321          1     43.62\n",
              "4      0004aac84e0df4da2b147fca70cf8255      288          1    196.89\n",
              "...                                 ...      ...        ...       ...\n",
              "93336  fffcf5a5ff07b0908bd4e2dbc735a684      447          1   2067.42\n",
              "93337  fffea47cd6d3cc0a88bd621562a9d061      262          1     84.58\n",
              "93338  ffff371b4d645b6ecea244b27531430a      568          1    112.46\n",
              "93339  ffff5962728ec6157033ef9805bacc48      119          1    133.69\n",
              "93340  ffffd2657e2aad2907e67c3e9daecbeb      484          1     71.56\n",
              "\n",
              "[93341 rows x 4 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-90c90adf-ab3b-47ee-b4c5-6a08fe45aba6\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Cust_unique_Id</th>\n",
              "      <th>Recency</th>\n",
              "      <th>Frequency</th>\n",
              "      <th>Monetary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>\n",
              "      <td>111</td>\n",
              "      <td>1</td>\n",
              "      <td>141.90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>\n",
              "      <td>114</td>\n",
              "      <td>1</td>\n",
              "      <td>27.19</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0000f46a3911fa3c0805444483337064</td>\n",
              "      <td>537</td>\n",
              "      <td>1</td>\n",
              "      <td>86.22</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0000f6ccb0745a6a4b88665a16c9f078</td>\n",
              "      <td>321</td>\n",
              "      <td>1</td>\n",
              "      <td>43.62</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0004aac84e0df4da2b147fca70cf8255</td>\n",
              "      <td>288</td>\n",
              "      <td>1</td>\n",
              "      <td>196.89</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93336</th>\n",
              "      <td>fffcf5a5ff07b0908bd4e2dbc735a684</td>\n",
              "      <td>447</td>\n",
              "      <td>1</td>\n",
              "      <td>2067.42</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93337</th>\n",
              "      <td>fffea47cd6d3cc0a88bd621562a9d061</td>\n",
              "      <td>262</td>\n",
              "      <td>1</td>\n",
              "      <td>84.58</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93338</th>\n",
              "      <td>ffff371b4d645b6ecea244b27531430a</td>\n",
              "      <td>568</td>\n",
              "      <td>1</td>\n",
              "      <td>112.46</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93339</th>\n",
              "      <td>ffff5962728ec6157033ef9805bacc48</td>\n",
              "      <td>119</td>\n",
              "      <td>1</td>\n",
              "      <td>133.69</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93340</th>\n",
              "      <td>ffffd2657e2aad2907e67c3e9daecbeb</td>\n",
              "      <td>484</td>\n",
              "      <td>1</td>\n",
              "      <td>71.56</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>93341 rows × 4 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-90c90adf-ab3b-47ee-b4c5-6a08fe45aba6')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-90c90adf-ab3b-47ee-b4c5-6a08fe45aba6 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-90c90adf-ab3b-47ee-b4c5-6a08fe45aba6');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 56
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm.describe().round(3).style.background_gradient()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "usjTNRyq3BiT",
        "outputId": "4cf8ba1c-1581-4637-be86-10661edf14db"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<pandas.io.formats.style.Styler at 0x7fd81ec46150>"
            ],
            "text/html": [
              "<style type=\"text/css\">\n",
              "#T_ec955_row0_col0, #T_ec955_row0_col1, #T_ec955_row0_col2 {\n",
              "  background-color: #023858;\n",
              "  color: #f1f1f1;\n",
              "}\n",
              "#T_ec955_row1_col0, #T_ec955_row1_col1, #T_ec955_row1_col2, #T_ec955_row2_col0, #T_ec955_row2_col1, #T_ec955_row2_col2, #T_ec955_row3_col0, #T_ec955_row3_col1, #T_ec955_row3_col2, #T_ec955_row4_col0, #T_ec955_row4_col1, #T_ec955_row4_col2, #T_ec955_row5_col0, #T_ec955_row5_col1, #T_ec955_row5_col2, #T_ec955_row6_col0, #T_ec955_row6_col1, #T_ec955_row6_col2, #T_ec955_row7_col1 {\n",
              "  background-color: #fff7fb;\n",
              "  color: #000000;\n",
              "}\n",
              "#T_ec955_row7_col0 {\n",
              "  background-color: #fef6fb;\n",
              "  color: #000000;\n",
              "}\n",
              "#T_ec955_row7_col2 {\n",
              "  background-color: #e7e3f0;\n",
              "  color: #000000;\n",
              "}\n",
              "</style>\n",
              "<table id=\"T_ec955_\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr>\n",
              "      <th class=\"blank level0\" >&nbsp;</th>\n",
              "      <th class=\"col_heading level0 col0\" >Recency</th>\n",
              "      <th class=\"col_heading level0 col1\" >Frequency</th>\n",
              "      <th class=\"col_heading level0 col2\" >Monetary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row0\" class=\"row_heading level0 row0\" >count</th>\n",
              "      <td id=\"T_ec955_row0_col0\" class=\"data row0 col0\" >93341.000000</td>\n",
              "      <td id=\"T_ec955_row0_col1\" class=\"data row0 col1\" >93341.000000</td>\n",
              "      <td id=\"T_ec955_row0_col2\" class=\"data row0 col2\" >93341.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row1\" class=\"row_heading level0 row1\" >mean</th>\n",
              "      <td id=\"T_ec955_row1_col0\" class=\"data row1 col0\" >237.460000</td>\n",
              "      <td id=\"T_ec955_row1_col1\" class=\"data row1 col1\" >1.033000</td>\n",
              "      <td id=\"T_ec955_row1_col2\" class=\"data row1 col2\" >165.197000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row2\" class=\"row_heading level0 row2\" >std</th>\n",
              "      <td id=\"T_ec955_row2_col0\" class=\"data row2 col0\" >152.581000</td>\n",
              "      <td id=\"T_ec955_row2_col1\" class=\"data row2 col1\" >0.209000</td>\n",
              "      <td id=\"T_ec955_row2_col2\" class=\"data row2 col2\" >226.330000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row3\" class=\"row_heading level0 row3\" >min</th>\n",
              "      <td id=\"T_ec955_row3_col0\" class=\"data row3 col0\" >0.000000</td>\n",
              "      <td id=\"T_ec955_row3_col1\" class=\"data row3 col1\" >1.000000</td>\n",
              "      <td id=\"T_ec955_row3_col2\" class=\"data row3 col2\" >9.590000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row4\" class=\"row_heading level0 row4\" >25%</th>\n",
              "      <td id=\"T_ec955_row4_col0\" class=\"data row4 col0\" >114.000000</td>\n",
              "      <td id=\"T_ec955_row4_col1\" class=\"data row4 col1\" >1.000000</td>\n",
              "      <td id=\"T_ec955_row4_col2\" class=\"data row4 col2\" >63.050000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row5\" class=\"row_heading level0 row5\" >50%</th>\n",
              "      <td id=\"T_ec955_row5_col0\" class=\"data row5 col0\" >218.000000</td>\n",
              "      <td id=\"T_ec955_row5_col1\" class=\"data row5 col1\" >1.000000</td>\n",
              "      <td id=\"T_ec955_row5_col2\" class=\"data row5 col2\" >107.780000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row6\" class=\"row_heading level0 row6\" >75%</th>\n",
              "      <td id=\"T_ec955_row6_col0\" class=\"data row6 col0\" >346.000000</td>\n",
              "      <td id=\"T_ec955_row6_col1\" class=\"data row6 col1\" >1.000000</td>\n",
              "      <td id=\"T_ec955_row6_col2\" class=\"data row6 col2\" >182.540000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th id=\"T_ec955_level0_row7\" class=\"row_heading level0 row7\" >max</th>\n",
              "      <td id=\"T_ec955_row7_col0\" class=\"data row7 col0\" >695.000000</td>\n",
              "      <td id=\"T_ec955_row7_col1\" class=\"data row7 col1\" >15.000000</td>\n",
              "      <td id=\"T_ec955_row7_col2\" class=\"data row7 col2\" >13664.080000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "(rfm[rfm['Frequency']>1].shape[0]/96095)*100 "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SWneUSry357V",
        "outputId": "0213a516-03f7-400c-9dac-110a5138b601"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2.913783235340028"
            ]
          },
          "metadata": {},
          "execution_count": 58
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Berdasarkan Exploratory Data menyatakan bahwa 2,9% merupakan customer yang berbelanja secara berulang sedangkan 97,1% adalah customer yang melakukan 1 kali transaksi dalam 1 waktu"
      ],
      "metadata": {
        "id": "hCE9-fv6qzyb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### RFM Analysis"
      ],
      "metadata": {
        "id": "lf_SPttk3dXz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "quantiles = rfm.quantile(q=[0.25,0.5,0.75])\n",
        "quantiles.to_dict()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QUl5wF9M3hBx",
        "outputId": "b8783625-a5b7-47e0-fa12-4ac487f83ba2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Recency': {0.25: 114.0, 0.5: 218.0, 0.75: 346.0},\n",
              " 'Frequency': {0.25: 1.0, 0.5: 1.0, 0.75: 1.0},\n",
              " 'Monetary': {0.25: 63.05, 0.5: 107.78, 0.75: 182.54}}"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)\n",
        "def RScore(x,p,d):\n",
        "    if x <= d[p][0.25]:\n",
        "        return 4\n",
        "    elif x <= d[p][0.50]:\n",
        "        return 3\n",
        "    elif x <= d[p][0.75]: \n",
        "        return 2\n",
        "    else:\n",
        "        return 1\n",
        "# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)\n",
        "def FMScore(x,p,d):\n",
        "    if x <= d[p][0.25]:\n",
        "        return 1\n",
        "    elif x <= d[p][0.50]:\n",
        "        return 2\n",
        "    elif x <= d[p][0.75]: \n",
        "        return 3\n",
        "    else:\n",
        "        return 4"
      ],
      "metadata": {
        "id": "3YCaF4ts3uBM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rfm_segmentation = rfm\n",
        "rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))\n",
        "rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))\n",
        "rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))"
      ],
      "metadata": {
        "id": "OmH3KH-J49Sw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \\\n",
        "                            + rfm_segmentation.F_Quartile.map(str) \\\n",
        "                            + rfm_segmentation.M_Quartile.map(str)\n",
        "rfm_segmentation"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "id": "ikS8rH3u493U",
        "outputId": "07609371-143f-47e0-c055-ef6d44b1c56e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                         Cust_unique_Id  Recency  Frequency  Monetary  \\\n",
              "0      0000366f3b9a7992bf8c76cfdf3221e2      111          1    141.90   \n",
              "1      0000b849f77a49e4a4ce2b2a4ca5be3f      114          1     27.19   \n",
              "2      0000f46a3911fa3c0805444483337064      537          1     86.22   \n",
              "3      0000f6ccb0745a6a4b88665a16c9f078      321          1     43.62   \n",
              "4      0004aac84e0df4da2b147fca70cf8255      288          1    196.89   \n",
              "...                                 ...      ...        ...       ...   \n",
              "93336  fffcf5a5ff07b0908bd4e2dbc735a684      447          1   2067.42   \n",
              "93337  fffea47cd6d3cc0a88bd621562a9d061      262          1     84.58   \n",
              "93338  ffff371b4d645b6ecea244b27531430a      568          1    112.46   \n",
              "93339  ffff5962728ec6157033ef9805bacc48      119          1    133.69   \n",
              "93340  ffffd2657e2aad2907e67c3e9daecbeb      484          1     71.56   \n",
              "\n",
              "       R_Quartile  F_Quartile  M_Quartile RFMScore  \n",
              "0               4           1           3      413  \n",
              "1               4           1           1      411  \n",
              "2               1           1           2      112  \n",
              "3               2           1           1      211  \n",
              "4               2           1           4      214  \n",
              "...           ...         ...         ...      ...  \n",
              "93336           1           1           4      114  \n",
              "93337           2           1           2      212  \n",
              "93338           1           1           3      113  \n",
              "93339           3           1           3      313  \n",
              "93340           1           1           2      112  \n",
              "\n",
              "[93341 rows x 8 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-413c37a5-36e6-4ae6-8f85-320c534d8c4a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Cust_unique_Id</th>\n",
              "      <th>Recency</th>\n",
              "      <th>Frequency</th>\n",
              "      <th>Monetary</th>\n",
              "      <th>R_Quartile</th>\n",
              "      <th>F_Quartile</th>\n",
              "      <th>M_Quartile</th>\n",
              "      <th>RFMScore</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>\n",
              "      <td>111</td>\n",
              "      <td>1</td>\n",
              "      <td>141.90</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>413</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>\n",
              "      <td>114</td>\n",
              "      <td>1</td>\n",
              "      <td>27.19</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>411</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0000f46a3911fa3c0805444483337064</td>\n",
              "      <td>537</td>\n",
              "      <td>1</td>\n",
              "      <td>86.22</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>112</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0000f6ccb0745a6a4b88665a16c9f078</td>\n",
              "      <td>321</td>\n",
              "      <td>1</td>\n",
              "      <td>43.62</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>211</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0004aac84e0df4da2b147fca70cf8255</td>\n",
              "      <td>288</td>\n",
              "      <td>1</td>\n",
              "      <td>196.89</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>214</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93336</th>\n",
              "      <td>fffcf5a5ff07b0908bd4e2dbc735a684</td>\n",
              "      <td>447</td>\n",
              "      <td>1</td>\n",
              "      <td>2067.42</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>114</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93337</th>\n",
              "      <td>fffea47cd6d3cc0a88bd621562a9d061</td>\n",
              "      <td>262</td>\n",
              "      <td>1</td>\n",
              "      <td>84.58</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>212</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93338</th>\n",
              "      <td>ffff371b4d645b6ecea244b27531430a</td>\n",
              "      <td>568</td>\n",
              "      <td>1</td>\n",
              "      <td>112.46</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>113</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93339</th>\n",
              "      <td>ffff5962728ec6157033ef9805bacc48</td>\n",
              "      <td>119</td>\n",
              "      <td>1</td>\n",
              "      <td>133.69</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>313</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93340</th>\n",
              "      <td>ffffd2657e2aad2907e67c3e9daecbeb</td>\n",
              "      <td>484</td>\n",
              "      <td>1</td>\n",
              "      <td>71.56</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>112</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>93341 rows × 8 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-413c37a5-36e6-4ae6-8f85-320c534d8c4a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-413c37a5-36e6-4ae6-8f85-320c534d8c4a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-413c37a5-36e6-4ae6-8f85-320c534d8c4a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Segmentations for customers are as follows\n",
        "best_customer = rfm_segmentation.loc[rfm_segmentation['RFMScore']=='444']\n",
        "loyal_customer = rfm_segmentation.loc[rfm_segmentation['F_Quartile']==4]\n",
        "big_spenders = rfm_segmentation.loc[rfm_segmentation['M_Quartile']==4]\n",
        "almost_lost = rfm_segmentation.loc[rfm_segmentation['RFMScore']=='244']\n",
        "lost_customer = rfm_segmentation.loc[rfm_segmentation['RFMScore']=='144']\n",
        "lost_cheap = rfm_segmentation.loc[rfm_segmentation['RFMScore']=='111']\n",
        "\n",
        "print(\"Best Customers: \", len(best_customer))\n",
        "print(\"Loyal Customers: \", len(loyal_customer))\n",
        "print(\"Big Spenders: \", len(big_spenders))\n",
        "print(\"Almost Lost: \", len(almost_lost))\n",
        "print(\"Lost Customers: \", len(lost_customer))\n",
        "print(\"Lost Cheap Customers: \", len(lost_cheap))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7YueW-RI6U_0",
        "outputId": "be38166e-22a0-4c03-cbb9-4c2d2dfbaca2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best Customers:  524\n",
            "Loyal Customers:  2800\n",
            "Big Spenders:  23335\n",
            "Almost Lost:  422\n",
            "Lost Customers:  337\n",
            "Lost Cheap Customers:  5954\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_temp = df_temp.merge(recency, on = 'customer_unique_id')\n",
        "df_temp = df_temp.merge(frequency, on = 'customer_unique_id')\n",
        "df_temp = df_temp.merge(monetary, on = 'customer_unique_id')\n",
        "df_temp.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 299
        },
        "id": "MhVSnh2UtjqE",
        "outputId": "a3c34069-e0dc-48fd-e741-ef5073bb4440"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                           order_id                     customer_id_x  \\\n",
              "0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "1  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "2  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "3  69923a4e07ce446644394df37a710286  31f31efcb333fcbad2b1371c8cf0fa84   \n",
              "4  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef   \n",
              "\n",
              "  order_status_x order_purchase_timestamp_x order_approved_at_x  \\\n",
              "0      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "1      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "2      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "3      delivered        2017-09-04 11:26:38 2017-09-04 11:43:25   \n",
              "4      delivered        2018-07-24 20:41:37 2018-07-26 03:24:27   \n",
              "\n",
              "  order_delivered_carrier_date_x order_delivered_customer_date_x  \\\n",
              "0            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "1            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "2            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "3            2017-09-04 21:22:15             2017-09-05 19:20:20   \n",
              "4            2018-07-26 14:31:00             2018-08-07 15:27:45   \n",
              "\n",
              "  order_estimated_delivery_date_x                customer_unique_id  \\\n",
              "0                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "1                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "2                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "3                      2017-09-15  7c396fd4830fd04220f754e42b4e5bff   \n",
              "4                      2018-08-13  af07308b275d755c9edb36a90c618231   \n",
              "\n",
              "   customer_zip_code_prefix  ... order_status_y order_purchase_timestamp_y  \\\n",
              "0                      3149  ...      delivered        2017-10-02 10:56:33   \n",
              "1                      3149  ...      delivered        2017-10-02 10:56:33   \n",
              "2                      3149  ...      delivered        2017-10-02 10:56:33   \n",
              "3                      3149  ...      delivered        2017-09-04 11:26:38   \n",
              "4                     47813  ...      delivered        2018-07-24 20:41:37   \n",
              "\n",
              "   order_approved_at_y order_delivered_carrier_date_y  \\\n",
              "0  2017-10-02 11:07:15            2017-10-04 19:55:00   \n",
              "1  2017-10-02 11:07:15            2017-10-04 19:55:00   \n",
              "2  2017-10-02 11:07:15            2017-10-04 19:55:00   \n",
              "3  2017-09-04 11:43:25            2017-09-04 21:22:15   \n",
              "4  2018-07-26 03:24:27            2018-07-26 14:31:00   \n",
              "\n",
              "   order_delivered_customer_date_y  order_estimated_delivery_date_y  \\\n",
              "0              2017-10-10 21:25:13                       2017-10-18   \n",
              "1              2017-10-10 21:25:13                       2017-10-18   \n",
              "2              2017-10-10 21:25:13                       2017-10-18   \n",
              "3              2017-09-05 19:20:20                       2017-09-15   \n",
              "4              2018-08-07 15:27:45                       2018-08-13   \n",
              "\n",
              "  order_purchase_timestamp recent_days order_approved_at payment_value_y  \n",
              "0               2017-10-02         331                 2           82.82  \n",
              "1               2017-10-02         331                 2           82.82  \n",
              "2               2017-10-02         331                 2           82.82  \n",
              "3               2017-10-02         331                 2           82.82  \n",
              "4               2018-07-24          36                 1          141.46  \n",
              "\n",
              "[5 rows x 27 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e8c91cd0-4311-480a-8ee2-59c3258c8794\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_id</th>\n",
              "      <th>customer_id_x</th>\n",
              "      <th>order_status_x</th>\n",
              "      <th>order_purchase_timestamp_x</th>\n",
              "      <th>order_approved_at_x</th>\n",
              "      <th>order_delivered_carrier_date_x</th>\n",
              "      <th>order_delivered_customer_date_x</th>\n",
              "      <th>order_estimated_delivery_date_x</th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>customer_zip_code_prefix</th>\n",
              "      <th>...</th>\n",
              "      <th>order_status_y</th>\n",
              "      <th>order_purchase_timestamp_y</th>\n",
              "      <th>order_approved_at_y</th>\n",
              "      <th>order_delivered_carrier_date_y</th>\n",
              "      <th>order_delivered_customer_date_y</th>\n",
              "      <th>order_estimated_delivery_date_y</th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "      <th>recent_days</th>\n",
              "      <th>order_approved_at</th>\n",
              "      <th>payment_value_y</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>69923a4e07ce446644394df37a710286</td>\n",
              "      <td>31f31efcb333fcbad2b1371c8cf0fa84</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-09-04 11:26:38</td>\n",
              "      <td>2017-09-04 11:43:25</td>\n",
              "      <td>2017-09-04 21:22:15</td>\n",
              "      <td>2017-09-05 19:20:20</td>\n",
              "      <td>2017-09-15</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-09-04 11:26:38</td>\n",
              "      <td>2017-09-04 11:43:25</td>\n",
              "      <td>2017-09-04 21:22:15</td>\n",
              "      <td>2017-09-05 19:20:20</td>\n",
              "      <td>2017-09-15</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>53cdb2fc8bc7dce0b6741e2150273451</td>\n",
              "      <td>b0830fb4747a6c6d20dea0b8c802d7ef</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "      <td>af07308b275d755c9edb36a90c618231</td>\n",
              "      <td>47813</td>\n",
              "      <td>...</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "      <td>2018-07-24</td>\n",
              "      <td>36</td>\n",
              "      <td>1</td>\n",
              "      <td>141.46</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 27 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e8c91cd0-4311-480a-8ee2-59c3258c8794')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e8c91cd0-4311-480a-8ee2-59c3258c8794 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e8c91cd0-4311-480a-8ee2-59c3258c8794');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Churn"
      ],
      "metadata": {
        "id": "HZ6XX0-D5XmT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Membuat kolom churn : 365 Hari sejak pembelian terakhir tidak order lagi\n",
        "\n",
        "df_temp['is_churn'] = df_temp['recent_days'].apply(lambda x: True if x >= 365 else False)\n",
        "df_temp['is_churn'].value_counts()\n",
        "df_temp.head()"
      ],
      "metadata": {
        "id": "pSP_eqlQrjSy",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 299
        },
        "outputId": "a202559a-718c-4759-e732-59b31bc9b601"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                           order_id                     customer_id_x  \\\n",
              "0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "1  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "2  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   \n",
              "3  69923a4e07ce446644394df37a710286  31f31efcb333fcbad2b1371c8cf0fa84   \n",
              "4  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef   \n",
              "\n",
              "  order_status_x order_purchase_timestamp_x order_approved_at_x  \\\n",
              "0      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "1      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "2      delivered        2017-10-02 10:56:33 2017-10-02 11:07:15   \n",
              "3      delivered        2017-09-04 11:26:38 2017-09-04 11:43:25   \n",
              "4      delivered        2018-07-24 20:41:37 2018-07-26 03:24:27   \n",
              "\n",
              "  order_delivered_carrier_date_x order_delivered_customer_date_x  \\\n",
              "0            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "1            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "2            2017-10-04 19:55:00             2017-10-10 21:25:13   \n",
              "3            2017-09-04 21:22:15             2017-09-05 19:20:20   \n",
              "4            2018-07-26 14:31:00             2018-08-07 15:27:45   \n",
              "\n",
              "  order_estimated_delivery_date_x                customer_unique_id  \\\n",
              "0                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "1                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "2                      2017-10-18  7c396fd4830fd04220f754e42b4e5bff   \n",
              "3                      2017-09-15  7c396fd4830fd04220f754e42b4e5bff   \n",
              "4                      2018-08-13  af07308b275d755c9edb36a90c618231   \n",
              "\n",
              "   customer_zip_code_prefix  ... order_purchase_timestamp_y  \\\n",
              "0                      3149  ...        2017-10-02 10:56:33   \n",
              "1                      3149  ...        2017-10-02 10:56:33   \n",
              "2                      3149  ...        2017-10-02 10:56:33   \n",
              "3                      3149  ...        2017-09-04 11:26:38   \n",
              "4                     47813  ...        2018-07-24 20:41:37   \n",
              "\n",
              "  order_approved_at_y  order_delivered_carrier_date_y  \\\n",
              "0 2017-10-02 11:07:15             2017-10-04 19:55:00   \n",
              "1 2017-10-02 11:07:15             2017-10-04 19:55:00   \n",
              "2 2017-10-02 11:07:15             2017-10-04 19:55:00   \n",
              "3 2017-09-04 11:43:25             2017-09-04 21:22:15   \n",
              "4 2018-07-26 03:24:27             2018-07-26 14:31:00   \n",
              "\n",
              "  order_delivered_customer_date_y  order_estimated_delivery_date_y  \\\n",
              "0             2017-10-10 21:25:13                       2017-10-18   \n",
              "1             2017-10-10 21:25:13                       2017-10-18   \n",
              "2             2017-10-10 21:25:13                       2017-10-18   \n",
              "3             2017-09-05 19:20:20                       2017-09-15   \n",
              "4             2018-08-07 15:27:45                       2018-08-13   \n",
              "\n",
              "   order_purchase_timestamp recent_days order_approved_at payment_value_y  \\\n",
              "0                2017-10-02         331                 2           82.82   \n",
              "1                2017-10-02         331                 2           82.82   \n",
              "2                2017-10-02         331                 2           82.82   \n",
              "3                2017-10-02         331                 2           82.82   \n",
              "4                2018-07-24          36                 1          141.46   \n",
              "\n",
              "  is_churn  \n",
              "0    False  \n",
              "1    False  \n",
              "2    False  \n",
              "3    False  \n",
              "4    False  \n",
              "\n",
              "[5 rows x 28 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f7c90e1d-d731-4f0e-9b1d-10f1d1b61c1a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_id</th>\n",
              "      <th>customer_id_x</th>\n",
              "      <th>order_status_x</th>\n",
              "      <th>order_purchase_timestamp_x</th>\n",
              "      <th>order_approved_at_x</th>\n",
              "      <th>order_delivered_carrier_date_x</th>\n",
              "      <th>order_delivered_customer_date_x</th>\n",
              "      <th>order_estimated_delivery_date_x</th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>customer_zip_code_prefix</th>\n",
              "      <th>...</th>\n",
              "      <th>order_purchase_timestamp_y</th>\n",
              "      <th>order_approved_at_y</th>\n",
              "      <th>order_delivered_carrier_date_y</th>\n",
              "      <th>order_delivered_customer_date_y</th>\n",
              "      <th>order_estimated_delivery_date_y</th>\n",
              "      <th>order_purchase_timestamp</th>\n",
              "      <th>recent_days</th>\n",
              "      <th>order_approved_at</th>\n",
              "      <th>payment_value_y</th>\n",
              "      <th>is_churn</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>e481f51cbdc54678b7cc49136f2d6af7</td>\n",
              "      <td>9ef432eb6251297304e76186b10a928d</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>2017-10-02 10:56:33</td>\n",
              "      <td>2017-10-02 11:07:15</td>\n",
              "      <td>2017-10-04 19:55:00</td>\n",
              "      <td>2017-10-10 21:25:13</td>\n",
              "      <td>2017-10-18</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>69923a4e07ce446644394df37a710286</td>\n",
              "      <td>31f31efcb333fcbad2b1371c8cf0fa84</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2017-09-04 11:26:38</td>\n",
              "      <td>2017-09-04 11:43:25</td>\n",
              "      <td>2017-09-04 21:22:15</td>\n",
              "      <td>2017-09-05 19:20:20</td>\n",
              "      <td>2017-09-15</td>\n",
              "      <td>7c396fd4830fd04220f754e42b4e5bff</td>\n",
              "      <td>3149</td>\n",
              "      <td>...</td>\n",
              "      <td>2017-09-04 11:26:38</td>\n",
              "      <td>2017-09-04 11:43:25</td>\n",
              "      <td>2017-09-04 21:22:15</td>\n",
              "      <td>2017-09-05 19:20:20</td>\n",
              "      <td>2017-09-15</td>\n",
              "      <td>2017-10-02</td>\n",
              "      <td>331</td>\n",
              "      <td>2</td>\n",
              "      <td>82.82</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>53cdb2fc8bc7dce0b6741e2150273451</td>\n",
              "      <td>b0830fb4747a6c6d20dea0b8c802d7ef</td>\n",
              "      <td>delivered</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "      <td>af07308b275d755c9edb36a90c618231</td>\n",
              "      <td>47813</td>\n",
              "      <td>...</td>\n",
              "      <td>2018-07-24 20:41:37</td>\n",
              "      <td>2018-07-26 03:24:27</td>\n",
              "      <td>2018-07-26 14:31:00</td>\n",
              "      <td>2018-08-07 15:27:45</td>\n",
              "      <td>2018-08-13</td>\n",
              "      <td>2018-07-24</td>\n",
              "      <td>36</td>\n",
              "      <td>1</td>\n",
              "      <td>141.46</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 28 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f7c90e1d-d731-4f0e-9b1d-10f1d1b61c1a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f7c90e1d-d731-4f0e-9b1d-10f1d1b61c1a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f7c90e1d-d731-4f0e-9b1d-10f1d1b61c1a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# New Dataset Merge"
      ],
      "metadata": {
        "id": "7Dyj3jvM9CAR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Mengambil order id dari banyaknya pembelian oleh pelanggan\n",
        "customer_order = pd.merge(customers,orders[['order_id','customer_id','order_purchase_timestamp']],on='customer_id')"
      ],
      "metadata": {
        "id": "DAySHFx79DIr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# mencari total pengeluaran pesanan berdasarkan id pesanan yang sama\n",
        "payment = payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()\n",
        "# marge total pembayaran setiap pesanan ke pelanggan yang telah membelinya yang bertujuan untuk menenmukan jumlah total pembelian\n",
        "order =pd.merge(customer_order,payment,on='order_id')"
      ],
      "metadata": {
        "id": "OQaZAAXw9LrB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Dataset Preparation"
      ],
      "metadata": {
        "id": "_F8nHl0C7nY0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.merge(customers.drop(columns=['customer_zip_code_prefix']),orders[['customer_id','order_id','order_purchase_timestamp']],on='customer_id')\n",
        "df2 = pd.merge(df1,reviews[['order_id','review_score']],on='order_id')\n",
        "paid = payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()\n",
        "df3 = pd.merge(df2,paid,on='order_id')\n",
        "df3['order_purchase_timestamp']=pd.to_datetime(df3['order_purchase_timestamp']).dt.date"
      ],
      "metadata": {
        "id": "RCxafDL17qv7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "number_of_days_for_purchase=180\n",
        "max_date_in_data= df3['order_purchase_timestamp'].max()\n",
        "data_split_date=max_date_in_data-dt.timedelta(days=number_of_days_for_purchase)"
      ],
      "metadata": {
        "id": "eyy5RmF996XG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_full = df3[df3['order_purchase_timestamp']<=data_split_date]\n",
        "df_last = df3[df3['order_purchase_timestamp']>data_split_date]"
      ],
      "metadata": {
        "id": "_z_1zZws99t8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_last_180=pd.DataFrame({'customer_unique_id':df3['customer_unique_id'].values.tolist()})\n",
        "df_last_180=df_last_180.merge(df_last.groupby(['customer_unique_id'])['payment_value'].sum().reset_index(),how='outer',on='customer_unique_id')\n",
        "df_last_180.fillna(0,inplace=True)"
      ],
      "metadata": {
        "id": "YbK0X-HXAqbQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_last_180['purchased']=np.where(df_last_180['payment_value']>0, 1,0)\n",
        "df_last_180.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "byoaD0O7Atvr",
        "outputId": "8673ed9b-9f47-4eca-cf51-23329b077676"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                 customer_unique_id  payment_value  purchased\n",
              "0  861eff4711a542e4b93843c6dd7febb0           0.00          0\n",
              "1  290c77bc529b7ac935b93aa66c333dc3           0.00          0\n",
              "2  060e732b5b29e8181a18229c7b0b2b5e         157.73          1\n",
              "3  259dac757896d24d7702b9acbbff3f3c         173.30          1\n",
              "4  345ecd01c38d18a9036ed96c73b8d066         252.25          1"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f4be2911-3d17-44e1-bf9e-a9c6246edb62\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>customer_unique_id</th>\n",
              "      <th>payment_value</th>\n",
              "      <th>purchased</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>861eff4711a542e4b93843c6dd7febb0</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>290c77bc529b7ac935b93aa66c333dc3</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>060e732b5b29e8181a18229c7b0b2b5e</td>\n",
              "      <td>157.73</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>259dac757896d24d7702b9acbbff3f3c</td>\n",
              "      <td>173.30</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>345ecd01c38d18a9036ed96c73b8d066</td>\n",
              "      <td>252.25</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f4be2911-3d17-44e1-bf9e-a9c6246edb62')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f4be2911-3d17-44e1-bf9e-a9c6246edb62 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f4be2911-3d17-44e1-bf9e-a9c6246edb62');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 72
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# New Dataset Feature"
      ],
      "metadata": {
        "id": "h73K72ToBFGN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# jumlah total perpelanggan\n",
        "tot_Amount=df_full.groupby('customer_unique_id')['payment_value'].sum().reset_index().rename(columns={'payment_value':'total_amount'})\n",
        "# rata-rata ulasan yang diberikan\n",
        "avg_review=df_full.groupby('customer_unique_id')['review_score'].mean().reset_index().rename(columns={'review_score':'avg_review'})\n",
        "# pembelian pertama dengan pembelian hari ini\n",
        "min_max_date=df_full.groupby('customer_unique_id')['order_purchase_timestamp'].agg([min,max])\n",
        "min_max_date['diff_first_today']=(dt.datetime.today().date()-min_max_date['min']).dt.days\n",
        "# pembelian bulan pertama hingga terakhir\n",
        "min_max_date['max']=pd.to_datetime(min_max_date['max'])\n",
        "min_max_date['min']=pd.to_datetime(min_max_date['min'])\n",
        "min_max_date['diff_first_last']=(min_max_date['max']-min_max_date['min']).dt.days\n",
        "# penjualan terbaru\n",
        "max_date=df_full['order_purchase_timestamp'].max()\n",
        "\n",
        "min_max_date['recency']=(np.datetime64(max_date)-min_max_date['max'])/np.timedelta64(1, 'M')\n",
        "# banyaknya penjualan\n",
        "frequency=df_full.groupby('customer_unique_id')['order_id'].count().reset_index().rename(columns={'order_id':'frequency'})"
      ],
      "metadata": {
        "id": "dVNKzvmDBIQs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#penggabungan features Engineering\n",
        "dataset=pd.merge(tot_Amount,avg_review,on='customer_unique_id')\n",
        "dataset=pd.merge(dataset,min_max_date,on='customer_unique_id')\n",
        "dataset=pd.merge(dataset,frequency,on='customer_unique_id')\n",
        "dataset=pd.merge(dataset,df_full[['customer_unique_id','customer_city','customer_state']],on='customer_unique_id')\n",
        "dataset.drop(['min','max'],axis=1,inplace=True)"
      ],
      "metadata": {
        "id": "7NmGqfuuBSMA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Exploratory Data Analysis (EDA)"
      ],
      "metadata": {
        "id": "YTbBS-d7EJaK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(12,10))\n",
        "plt.subplot(3, 1, 1); sns.distplot(rfm['Recency'],kde=False)\n",
        "plt.subplot(3, 1, 2); sns.distplot(rfm['Frequency'],kde=False)\n",
        "plt.subplot(3, 1, 3); sns.distplot(rfm['Monetary'],kde=False)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 661
        },
        "id": "hxEzitjy5m_u",
        "outputId": "d48a2517-7e1d-4e55-9f82-ee24d59c73ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
            "  warnings.warn(msg, FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 864x720 with 3 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm_segmentation.plot(kind = \"scatter\",\n",
        "                     x = \"Recency\",\n",
        "                     y = \"Monetary\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "z2l90JrV5xcu",
        "outputId": "34e03bd5-0ec6-4e08-be27-cc257af182c5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZEAAAEGCAYAAACkQqisAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deZxU5Zno8d9T1QubAgIu0CgaXC54FbUjGtQb9Y5BY9C5GuOSSBKX+dzRxEyioJM4bhMTzTYajRmiTmRiVIREETVeBI3RKLFRGgG3Doo0KGCzSCNUd1U994/zVlNdfWo7XWv38/18+kPVe07VeU9xznnOu5z3FVXFGGOMCSJU7gwYY4ypXhZEjDHGBGZBxBhjTGAWRIwxxgRmQcQYY0xgNeXOQKmNHDlSx40bV+5sGGNMVVm6dOnHqjoqNb3fBZFx48bR1NRU7mwYY0xVEZE1fulFq84SkftFZKOIrPBZ9j0RUREZ6d6LiNwpIi0islxEjk5ad7qIvOv+pielHyMib7jP3CkiUqx9McYY46+YbSK/BaamJorIWOA04IOk5NOBg93f5cA9bt29gBuAycCxwA0iMtx95h7gsqTP9diWMcaY4ipaEFHVF4DNPot+AcwAkh+VPwuYrZ5XgGEish/wBWChqm5W1S3AQmCqW7anqr6i3iP3s4Gzi7Uvxhhj/JW0d5aInAWsU9XmlEVjgLVJ71tdWqb0Vp/0dNu9XESaRKRp06ZNvdgDY4wxyUoWRERkEPCvwL+VapsJqjpLVRtVtXHUqB6dC4wxxgRUypLIZ4ADgWYReR9oAF4TkX2BdcDYpHUbXFqm9AafdGMqWlt7hOa1W2lrj5Q7K8YURMm6+KrqG8DeifcukDSq6sciMh+4UkQexmtE36aqH4rIM8CtSY3ppwHXqepmEflERI4DlgAXA78s1b4YE8Tjy9Yxc95yakMhOuNxbj/nCKZNSlsLa0xVKGYX34eAl4FDRaRVRC7JsPpTwGqgBfgN8M8AqroZuAV41f3d7NJw69zrPvN34Oli7IcxhdDWHmHmvOXs6oyzPRJlV2ecGfOWW4nEVL2ilURU9YIsy8clvVbgijTr3Q/c75PeBBzeu1waUxqtW3ZSGwqxi3hXWm0oROuWnYwYUl/GnBnTOzZ2ljEl0DB8IJ3xeLe0znichuEDy5QjYwrDgogxJTBiSD23n3MEA2pD7FFfw4DaELefc4SVQkzV63djZxlTLtMmjWHK+JG0btlJw/CBFkBMn2BBxJgSGjGk3oKH6VOsOssYY0xgFkSMMcYEZkHEGGNMYBZEjDHGBGZBxBhjTGAWRIwxxgRmQcQYY0xgFkSMMcYEZkHEGGNMYBZEjDHGBGZBxBhjTGAWRIwxxgRmQcQYY0xgFkSMMcYEZkHEGGNMYBZEjDHGBFa0ICIi94vIRhFZkZT2ExF5S0SWi8gfRWRY0rLrRKRFRN4WkS8kpU91aS0icm1S+oEissSlPyIidcXaF2OMMf6KWRL5LTA1JW0hcLiqHgG8A1wHICITgPOBie4zvxKRsIiEgbuB04EJwAVuXYDbgF+o6nhgC3BJEffFGGOMj6IFEVV9Adickvb/VDXq3r4CNLjXZwEPq2pEVd8DWoBj3V+Lqq5W1Q7gYeAsERHgFGCu+/wDwNnF2hdjjDH+ytkm8k3gafd6DLA2aVmrS0uXPgLYmhSQEum+RORyEWkSkaZNmzYVKPvGGGPKEkRE5PtAFHiwFNtT1Vmq2qiqjaNGjSrFJo0xpl+oKfUGReTrwJnAqaqqLnkdMDZptQaXRpr0NmCYiNS40kjy+sYYY0qkpCUREZkKzACmqeqnSYvmA+eLSL2IHAgcDPwNeBU42PXEqsNrfJ/vgs9zwLnu89OBx0u1H8YYYzzF7OL7EPAycKiItIrIJcBdwB7AQhFZJiK/BlDVlcAcYBXwJ+AKVY25UsaVwDPAm8Acty7ATOC7ItKC10ZyX7H2xRhjjD/ZXaPUPzQ2NmpTU1O5s2GMMVVFRJaqamNquj2xbowxJjALIsYYYwKzIGKMMSYwCyLGGGMCsyBijDEmMAsixhhjArMgYowxJjALIsYYYwKzIGKMMSYwCyLGGGMCsyBijDEmMAsixhhjArMgYowxJjALIsYYYwKzIGKMMSYwCyLGGGMCsyBijDEmMAsixhhjArMgYowxJjALIsYYYwIrWhARkftFZKOIrEhK20tEForIu+7f4S5dROROEWkRkeUicnTSZ6a79d8VkelJ6ceIyBvuM3eKiBRrX4wxxvgrZknkt8DUlLRrgUWqejCwyL0HOB042P1dDtwDXtABbgAmA8cCNyQCj1vnsqTPpW7LGGNMkRUtiKjqC8DmlOSzgAfc6weAs5PSZ6vnFWCYiOwHfAFYqKqbVXULsBCY6pbtqaqvqKoCs5O+yxhjTImUuk1kH1X90L3+CNjHvR4DrE1ar9WlZUpv9Un3JSKXi0iTiDRt2rSpd3tgjDGmS9ka1l0JQku0rVmq2qiqjaNGjSrFJo0xpl8odRDZ4KqicP9udOnrgLFJ6zW4tEzpDT7pxhhjSqjUQWQ+kOhhNR14PCn9YtdL6zhgm6v2egY4TUSGuwb104Bn3LJPROQ41yvr4qTvMsYYUyI1xfpiEXkI+DwwUkRa8XpZ/RiYIyKXAGuA89zqTwFnAC3Ap8A3AFR1s4jcArzq1rtZVRON9f+M1wNsIPC0+zPGGFNC4jVN9B+NjY3a1NRU7mwYY0xVEZGlqtqYmm5PrBtjjAnMgogxxpjALIgYY4wJzIKIMcaYwCyIGGOMCcyCiDHGmMAsiBhjjAnMgogxxpjALIgYY4wJzIKIMcaYwCyIGGOMCcyCiDHGmMAsiBhjjAnMgogxxpjALIgYY4wJzIKIMcaYwCyIGGOMCcyCiDHGmMAsiBhjjAkspyAiIn8QkS+KiAUdY4wxXXINCr8CLgTeFZEfi8ihvdmoiPyLiKwUkRUi8pCIDBCRA0VkiYi0iMgjIlLn1q1371vc8nFJ33OdS39bRL7QmzwZY4zJX05BRFWfVdWLgKOB94FnReSvIvINEanNZ4MiMgb4NtCoqocDYeB84DbgF6o6HtgCXOI+cgmwxaX/wq2HiExwn5sITAV+JSLhfPJijDGmd3KunhKREcDXgUuB14E78ILKwgDbrQEGikgNMAj4EDgFmOuWPwCc7V6f5d7jlp8qIuLSH1bViKq+B7QAxwbIizHGmIBybRP5I/AXvAv+l1R1mqo+oqrfAobks0FVXQf8FPgAL3hsA5YCW1U16lZrBca412OAte6zUbf+iOR0n8+k5v9yEWkSkaZNmzblk11jjDEZZA0irjF9qapOUNUfqeqHyctVtTGfDYrIcLxSxIHAaGAwXnVU0ajqLFVtVNXGUaNGFXNTxhjTr2QNIqoaB84p4Db/N/Ceqm5S1U7gD8AUYJir3gJoANa51+uAsQBu+VCgLTnd5zPGGGNKINc2kUUico5ri+itD4DjRGSQ+75TgVXAc8C5bp3pwOPu9Xz3Hrd8saqqSz/f9d46EDgY+FsB8meMqUBt7RGa126lrT1S7qyYJDXZVwHgn4DvAlER2QUIoKq6Z74bVNUlIjIXeA2I4jXSzwKeBB4WkX93afe5j9wH/LeItACb8XpkoaorRWQOXgCKAleoaizf/BhjKt/jy9Yxc95yakMhOuNxbj/nCKZN8m0CNSUm3k19/9HY2KhNTU3lzoYxJkdt7RGm3LaYXZ3xrrQBtSFemnkKI4bUlzFn/YuILPVrA8+1JJJoED8YGJBIU9UXCpM9Y4zx17plJ7WhELvYHURqQyFat+y0IFIBcgoiInIpcBVe4/Uy4DjgZbxnO4wxpmgahg+kMx7vltYZj9MwfGCZcmSS5dqwfhXwWWCNqp4MHAVsLVqu+gFrJDQmNyOG1HP7OUcwoDbEHvU1DKgNcfs5R1gppELkWp21S1V3iQgiUq+qb/V2/Kz+zBoJjcnPtEljmDJ+JK1bdtIwfKAFkAqSaxBpFZFhwGPAQhHZAqwpXrb6rrb2CDPnLWdXZ7yrjnfGvOVMGT/STgxjMhgxpN7OkQqUUxBR1X90L28UkefwHvh7umi56sOskdAY05fkOnbWfydeq+qfVXU+cH/RctWHWSOhMaYvybVhfWLyGzfk+jGFz07flNyIbo2ExlQO6+DSexmrs0TkOuBf8YZt/wTvSXWADrynzE0W6RrRrZHQmPKyDi6FkdMT6yLyI1W9rgT5KbpSPrFuT9oaU5ns3MxfuifWc63O+r6IfFVErndfNlZEbAKoLBKN6MkSjejG9EXVUj1k52bh5NrF924gjveE+i1Au0v7bJHy1SdYI7rpT6qpesjOzcLJtSQyWVWvAHYBqOoWoK5oueojrBHd9BfJzz9tj0TZ1RlnxrzlFVsisXOzcHItiXS6HlkKICKjgHjmjxiwJ21N/1CNzz/ZuVkYuQaRO4E/AnuLyA/xJof6QdFy1cfYk7amr6vW6iE7N3svp+osVX0QmAH8CPgQOFtVHy1mxowx1cOqh/qvnOcTAd4FPkl8RkT2V9UPipIrY0zVseqh/inX+US+BdwAbABiuOlxgSOKlzVjTLWx6qH+J9eSyFXAoaraVszMGGOMqS65dvFdC2wrZkaMqQTV8rCcMZUi15LIauB5EXkS6Dq7VPXnQTbq5ia5Fzgcr1rsm8DbwCPAOOB94DxV3SIiAtwBnAF8CnxdVV9z3zOd3b3E/l1VHwiSH2Oguh6WM6ZS5FoS+QBYiPeA4R5Jf0HdAfxJVQ8DjgTeBK4FFqnqwcAi9x7gdOBg93c5cA+AiOyF104zGTgWuEFEhvciT6Yfq7aH5YypFLlOSnUTgIgMce/bg25QRIYCJwFfd9/VAXSIyFnA591qDwDPAzOBs4DZ6o0U+YqIDBOR/dy6C1V1s/vehcBU4KGgeTP9VzU+LGdMJch1UqrDReR1YCWwUkSWisjEbJ9L40BgE/BfIvK6iNwrIoOBfVT1Q7fOR8A+7vUYvDaZhFaXli7dL/+Xi0iTiDRt2rQpYLZNX1atD8sZU265VmfNAr6rqgeo6gHA94DfBNxmDXA0cI+qHgXsYHfVFQCu1JF9jPocqeosVW1U1cZRo0YV6mtNH2IPyxkTTK4N64NV9bnEG1V93pUegmgFWlV1iXs/Fy+IbBCR/VT1Q1ddtdEtXweMTfp8g0tbx+7qr0T68wHzZIw9LGdMALmWRFaLyPUiMs79/QCvx1beVPUjYK2IHOqSTgVWAfOB6S5tOvC4ez0fuFg8xwHbXLXXM8BpIjLcNaif5tKMCWzEkHqOHDvMAogxOcq1JPJN4CbgD+79X1xaUN8CHhSROrxg9A28gDZHRC4B1gDnuXWfwuve24LXxfcbAKq6WURuAV51692caGQ3xhhTGjlNj9uXlHJ6XGOM6SvSTY+bsSQiIvMzLVfVab3NmDHGmOqVrTrreLxutA8BS/AGXjTGGGOA7EFkX+AfgAuAC4EngYdUdWWxM2aMMabyZeydpaoxVf2Tqk4HjsNr3H5eRK4sSe6MMcZUtKy9s0SkHvgiXmlkHLunyjXGGNPPZWtYn4030u5TwE2quqIkuTLGGFMVspVEvoo3LMlVwLe9UdkBN7Ohqu5ZxLwZY4ypcBmDiKrm+kS7McaYfsiChDHGmMAsiBhjKoZNT1x9ch07y5RJW3vERpU1/YJNT1ydLIgUUKEv+HZSmf4ieXrixOySM+YtZ8r4kXbzVOEsiBRIoS/4dlKZ/sSmJ65e1iZSAMkX/O2RKLs641z9aDMtG7YH/s7ESZUscVIZ09fY9MTVy4JIAfhd8Dtiyhm/fJH5y9YF+s58TiprjDTVzqYnrl5WnVUAfhd8gI5oPHAVVOKkmpFSRZb6PdZuYvoKm564OlkQKYDEBf/qR5vpiHWf5Ks39brZTiprNzF9zYgh9XbsVhmrziqQaZPG8NS3T6SupvtP2tt63Uxzflu7ielrrGq2+lgQKaDx++zBT8/dXa9bXxPiis+PL9r2ytUYaSe6KYbHl61jym2L+eq9S5hy2+LA7YmmtCyIFNi0SWN4aeYpXHbSQYAy64XVRTshytEYaSe6KQa/Ho4z5i23G5UqULYgIiJhEXldRBa49weKyBIRaRGRR0SkzqXXu/ctbvm4pO+4zqW/LSJfKM+e+PvV8y1EolrUE6KtPcIBIwaz4MoT+N2lk3lp5im+jeqFKjnYiW6KpbdVs4U8xq2UnZ9yNqxfBbwJJIaTvw34hao+LCK/Bi4B7nH/blHV8SJyvlvvKyIyATgfmAiMBp4VkUNUNVbqHUlVigen/HplHTl2WE7rBe29ZQ+EmWLpTdVsoY5x6+kYTFlKIiLSgDdb4r3uvQCnAHPdKg8AZ7vXZ7n3uOWnuvXPAh5W1Yiqvoc3de+xpdmDzIrdVpFriaDQJQd7IMwUS9Cq2UId41bKDq5c1Vn/AcyArlvaEcBWVY26961A4hZgDLAWwC3f5tbvSvf5TDcicrmINIlI06ZNmwq5H76K3VaRa9G/0L23/Pbr+jMn0LplZ9FONqte6D8S7YmZqmZTFeoYt56OwZW8OktEzgQ2qupSEfl8KbapqrOAWQCNjY2aZfWCKOaDU7mWCIpRckjerxXrtnHLglVFK/5b9UL/k+9zIoU4xtvaI2zb2UlHzErZQZSjJDIFmCYi7wMP41Vj3QEME5FEUGsAEt1+1gFjAdzyoUBbcrrPZypCpmc8evu9uZR0ilUiGjGknobhA7nlyVVFK/5b9ULfUqwSZW+P8URvwysefI1YPE5tWGzYlTyVvCSiqtcB1wG4ksjVqnqRiDwKnIsXWKYDj7uPzHfvX3bLF6uqish84Pci8nO8hvWDgb+Vcl/KKdeSTrFKRMVuZLdG/L6j2CXKfI/xxJQNg+vCPUZ8qK+Buy86iomjh9pxlqNKGvZkJvCwiPw78Dpwn0u/D/hvEWkBNuP1yEJVV4rIHGAVEAWuqISeWaWUa9G/N0NJpJsjpdiN7NaIn59KnbysVEPz5HqMJwe0SDRGKCTdlteFwwwdWFdRv2GlK2sQUdXngefd69X49K5S1V3Al9N8/ofAD4uXw/4t0x1krgNEBlXs7+9LKrntqJJKlH4BjZSx7uxGJX+VVBLp1yrtTjKXO8hCVZWl23cb1TW7Sh+Es5JKlH4BrT4sqAj1YbtRCcqCSC8V4uJfiXeSud5B9nbU1Wz7XqhRXSstSBdKJd3p+6mkEqVfQJOQ8OSVJ7CjI9bnjo1SsSDSC4W4+Ae9k0y+KAIFv0CW4g6yVHfRlRikC6WS7vTTqZQSZbqANn6fPcqSn77CgkhA+V4A090JB7mTTL4o7uyMIiIMqAl3nRRBeqqkrluKO0i/fQ+HpKB30UECVTWVWirpTj+TSpknpFICWl9iQSSgfC7+me6E872TbGuPMGNuM5GoJm1b6Yx5D/t/79FmQuL1Msl2153tDr3YJ5zfvu+IxFixbpvvOGBB5BukC1lqKXZpMcEujPmplIDWV1gQCSjXi3+2O+F87yQfXPIBkWj6h+47XW+TSDTaY1u55GvCfnt2qx8u5gk3Ykg91585ge//cUW39FueXMXUw/ctyHbznau+UNVrycFoVzSGqjKwtqZo1WnF6MZdrM8FkbqtbAG6t3mrptJouVkQCWjEkHrOO6aB2a980JV2XmNDjwMulzvhXO8k29oj3P1cS175THfX7ZcvjStn/PLFbj1Vcr3YBT3pDh89lMF1YXZ07H7Ep5ANw/kE6UI1Uvt2JQW2RzIH9nIIWvIqZTtT6rbOO6aBOUtb0wZohV7lrS+3oRWDBZGA2tojzFna2i1tTlMrV516SKCH8nK5k2zdspO6cIhItPv3hUMwqLaGjlicWDxO8uJ0d90NwweyK9r92cxITAGlI5rfXXhvTrqG4QOJaXH76ucapPMttaT7Pr9glKxSek/1plNHqboV+20rcePmF6Cvmbsc0G7VvfnkrdK7TFcim9kwoFxH/cxnbJ9s4wv5XeTqa0I8c9VJ3H3RUfzm4kZumnZ4zuMIqWYeizJ5f9LlrbdjXBV7xOPk7WQbxyxRukzmV7rMNruj3/9TskrpPRV05Np8PtfbMbP8tpVJOCSEJfhovDaab/6sJBJQPnetudwJ53I3n65qZsl7m7lpwSrqwkI0rlx/5gQOHz20x7aS755bt+ykJhwimsPFLlPeClEFVMyG4WzVbMnjKK3ftpNHmtZ2W55auszlTjX1/8mvyqUS7mozHcOZ2hxyPfZTj5t0x2W2PKaOrptJLO6VprPlLdP2Kr3LdKWxIBJQ8oUiHBI6Y97FO91FO1N1VT5F6NQL7p9WfMT3H/MapjvcbCy3LFjFSzNP6fbZ1BP6u//7EHZ1+p+cA2q9O7HbzzkCIGPeCnXSFaMBP3mfO2Ixrjz5YC6cvH/XdhLLNa5EYkpdOERHyjAYqQEx16CZ+v+U+GwlNdSmuyl5seXjrJ0Cbj/nCK6Zu5yQCDEXIFKP/dTj5vt/XMHgujAx1ZyrPF9s+ZhY0vFVGxYuOHYsc5rSt4kAgbs8V0uX6UpiQaQXpk0aw/ZdUW56YiW14RC3LFjFHvU1TJs0Jq92gnzv5hMX3Lb2CDc9sbLH8nBIWLl+G0MH1nVdwFK7Bf904TvUhaDDJ47E48pT3z6R8fvsQfParRnzVqknnd9F7GcL3+Gu51r4ybneszSJ5Ql+d7ypATGfoJkaGMv9m/jxC3ZTbluctVOAAtFYvGvoqRvnr+w69iF9u1CiA0U+bS/JbXwhgatOPYSrTj0kY4DuTcnWukznx4JIL7S1R7jlyVV0xJSO2O6TY8J+e+bVOBf0br51y05EeqZHOuNcNruJWndnfeLBI3p0CxZARUgt+gPU1+zuLZVL3hIn3cr1nwDKxNFDM+a7FNJdxCJRr81m1tcaMzZ+D6oNE0d7BMRKDZq9kRzsmtduzbhubSjEyvWfMGNuc7exCztjyjVzM5dQU78nW5Vn65adaDzl+FQv3a99K9FuUYiu6fYsSe4siPSC34UqJMKLLZvyLlkEuTANrgv7PjMSjyvROERc76tFb/acEjgSjfOvpx/Gzxa+3eM7koNErnlLrgKphG6RmS5iXsOppl1eXxPi1187homj9/T9P6jEoAn+7T/5dr0eXBdOW80JuN9MXeN19959yaMNpFb37ojEenxPtpukwXVh12Nwt0hMGVwX7pZmXXLLy4JIL/hdqD7tiHHrU2+idC8idMQynzRBitA7OmLUh6XHiZZLM2RtCCYfNIK/Xnsq1z+2gqdWfLQ7L0eO7nZXly1vldgtMnERu8ZV4yXrjMeZOHpo10UuHlc6Ykp9TQhV5ZtTxqUNIAnZgmahH1bL9n1+F9Igz0ukO6aSS2YTRw8lpj2PslhcfUuoXVMpP7kqr5ukHR0xBtSGugW1AbWhbs8UVeKx199YEMlTcm+eHR0x/unEg7hzcUu3SqGOGNSElNqwdD1BHovHeanl44wncboidLqndQfXhZGQ9JgTIRehkHSd8Ivf3tht2ZymVp5640Oi8d0NoJmK97m06ZTjCeDERez3Sz7grudaqEsZ7rurTWvBKurCdD1/c8+fVzPrhdXcfNbhXHTcAT2+N9uFq9B3xtm+zy8/18xtBoRINL+La8PwgT2OKb+S2U/OPZLvzlnW1V5RGxZ+cq7/FM2J7tVTD983r2PA76ZLVdm2s4O29ggjhtRX/CjG/YEFkTwkTmaAXZ1x/FsUPN4Fa/fSaNw7saeMHwnQLRBlOqkyPa3bGY9zXmMDc5paCSF82pnbxI5hgX87cyIjhtT7NpwDtLvqh2vmLmfYoLqMd+bZ2k0efGVNty7IpapuSASuCyfvz4WT9/et6rnlyVVdD1cmiylerzeBiyZ3DySZLlyQuTdbkH1I/b6r53rtbonRZ1eu30YopXEsLCFSCsM5XVzTVV+edMiobuv5Vell27/kHm7J73PNy87OKHGFKx58vdtgo9Ylt7wsiOQo+WROyHT/3xGLE3Z17wmRqHL9Yyu67vx3dcapDwsSEt+7S68Bc3m3u8nUp3UfebWVn335CED43qPNPZ5mT6gJeRfD3y9ZQ004zE1PrGLl+m38n6PGsLMzmnY/ItE4/zS7iZgqN3xpou+deaZ2kwdfWdOjC3LqRTCIINU7qYEr25PlADc9sYqpE7uP45UpaGYLMPmWxvy+ryMa54w7/8JPv3wkyu6ed8liGgftHkVyvbjmWrU6Ykh9j+CSSZASWnKwumx2E5FovGuw0RnzlvPSzFO6HXsdsThXfH58znlKx8bOyp1ke2q5r2lsbNSmpqa8P9e8ditfvXdJV1fHbL7y2QYeebU1+4rOgNpQ17MdiZNNgJ0ZGjkTEvXV5zXu3mZqMBlUGyISU/cwVnchAZ9kX1edMp5/Oe1Q32V+1W7H/2hRj2cvAOrCwk+/fGSgEkni96kJedU1l554IJeecFC3Ekaim2pC8u+bnN/U9VINrA3xn187hpMO2bvb/r3U8nGPoDlt0pi0277+ixN6tAmk2/fUB/3S5bG+xrtJSQ0g9TUhfnKu//MSpWxwzrYffv8n6TSv3cpF977SVUIGGFwf5qYvTeTkw/YGvMFJ737u3ZxGsM7EGur9ichSVW3skW5BJDdt7RE+9+PFae/0UyW3h+RicF2Ym6ZNZNLYYZx514sZL2zphMULCOFQiF055jOd+pqeY3QlnNfYwO3nHpn1O5rXbuXC37zSrSE0WbaLSGr7U8PwgWzZ0cEZd/6l50OBYeFnLij5Bfw96mv43aWTewwxP3/ZOmYkVVH6qa8RvvLZ3Q+4dcbjXP/FCYzdayCf7Iyy58CaruqctvZIjzaY68+cwC0LVuV0AfW7gIFXekutdhtYGyIeh0jS8y2DasP8+mvHdJUQijWabb4lwSs+P55ZL6zO6f/ET3KJNlni4cV8fuNs+9ubYNeXpQsiJa/OEpGxwGxgH7y6nlmqeoeI7AU8AowD3gfOU9UtIiLAHcAZwKfA11X1Nfdd04EfuK/+d1V9oFj5HjGknitPHs/PFr6T0/r5BBDweqLc+MRK78IdMLDH1PvL1D8/V/84aTQPN/mXpOY0tXL5iQdlrY5qGD6QaIYiTi7zesDuahzHy4oAABcuSURBVD9voEbx/c7k5xT8qpsisXhX19DkC+Du6pJtLH5zI7NfWdOjVBaJKrNf7l6N+P3HVnQF2vqwgMCp/2NvFr25kdpwiLgqXz1+f845qoFla7cSTmmz8Nv3dA32L808hae+dUKP4OlXSo2jTBy9Z9f71A4RLRu2s2ztViaNHZb1/y/xfxBCiMTiTB43jCnjR1ETDvGLZ9/Jq6H/l4vf7fG75lq9lmi78pO4QbnpiVXUhrL/xtmUYqK0vqYcbSJR4Huq+pqI7AEsFZGFwNeBRar6YxG5FrgWmAmcDhzs/iYD9wCTXdC5AWjEC0ZLRWS+qm4pVsYvnLw/dz33bsb5PLIJu44viZJKXVi6LgztkdwaxoutvibEvNcyV8Xd83wLP/vKURnXSR2yIlVHLJZ1Xo+E3V1O0//2IZGuB9ES9eTgOkGocuZdL3Z1REjtCuvXrpBNoqSWyNtTb2xw6d7/4z3Pr+bev7xHfU2oR2ksdd/b2iM80bweSclCWITn3trIyYftzU+/fKT33IWIb+muvka62qL8Sgr/9tgb3aYuuPj4/bn5rP/pu29+/wd/Xb2Fv67efXql6ziwcv0nhHp0cd+9YzUhIRySnLr5QpqHDlPUhKAzZcQBv3HAsm2vFBOl9TUlH8VXVT9MlCRUdTvwJjAGOAtIlCQeAM52r88CZqvnFWCYiOwHfAFYqKqbXeBYCEwtZt5HDKnnJ+ceSX2Nz2PiGZw+ce+u14lzSYC5/3Qct/7j/2RIfdj/g2XSGYv36O2Tat7r65n157+nXe43ZEWqK08+2Pekznfk1oRY0p3ttEljWHDlCcTdxScSU3Z1xpn98gfdRhy++tFlrvNCcap1O2Pqe3MQV3ip5WPAu+M/7keLuPGJVexM+cESJdQpty0G4KWZp3DTtIk9jplBdWF+c3Fj15A7qaMMt2zY3i2AAMx++QNaNmz3zXfrlp09AkE64ZAX6NraIzy+bB2XzW7K2FMwGteu/5dc+D10mOrTjjhnHzWmx2jQL7Z8nHHE5VQjhngTpaW68YmVgUci7uvKOhS8iIwDjgKWAPuo6odu0Ud41V3gBZjkoVVbXVq6dL/tXC4iTSLStGlTz6e38zFt0hh+c3EjNaHcA8nTKzf2SAuHvC65Jx+2d8Yqn3IIiRDNoTru1qff4sFX1vRIb9mwnQf++n7Gg6s2LBw4cjAvvLORlg3beeGdjbzwziba2iNZh8xI5+yjxnQLSjs6YtTXZA7QHbHsQ+KDV4IspM6YMmPeclo2bGfG3OUZqz/bI7GuIfYBTj5sb3alXKQ7ojEmjh6admj+F13ASrUszTAnDcMHdmtryWRHJMYN81fyuR8v4uoMPQSTdcaV7z3aTFt7JOtw8YmHDpP5nX/zm9ez4MoT+N2lk3lp5indxkdLN02B37bH+pSOO2PKvX9ZnXW/+qOydfEVkSHAPOA7qvqJJN35qqqKpBbsg1PVWcAs8BrWe/t9o4dmruvPxc7OOJc+0MQlJxzIN6aM4zcvrM54115KA2vD/K9DRrLgjY+yrnvjEysYu9fArqE/fvDYCp5ekf1z0Zhy5UOv90ivCcHPz5vkO21uNvOb1zNz6mFdgcQbRjx7FaFf77FUMd1dFVko4ZB47SVpbkhSe81pXLuqZSRl3LPE+ZOui/HIIXW+25jkqmj8Rpw+9+gxPJKmXSxVus4TmTqYJC7M//XX9zP2hPKr8vQ7/2pDXrVhotop3ThgifaN9L2w/P8/7v3Lai498SBrG0lRliAiIrV4AeRBVf2DS94gIvup6oeuuipx+74OGJv08QaXtg74fEr688XMd8KOjlhBLigdMeWeP3t3N4meVaUulITFuxQlb3dXNMYBIwbl9PnOGFw2e6l3Uqvm/JukWy0ah395ZBnTjz+AELkN4ZLgDQ64e/TiF1s+LuzvKUJdCGrCITqiMWLx7vtRHxbieGOXDagNs6szlvH32BGJsXbzp77drqHnsZAYN6p1y07qa0J0JgXIATXhriDg9wzL8Z8ZycXH79/VQQC8NpHx++zB48vWMWOuN8ZVLK785FzvYjpj6mE5B5F0OmPKtCP3ZX6z/43FrBdWE9P07SsJJ3xmBM++lbkWoSMWZ9vOzq6n2f3GAdvV6XWwyDTqwMTRe/qe33XuN7Yg0l05emcJcB/wpqr+PGnRfGA68GP37+NJ6VeKyMN4DevbXKB5BrhVRIa79U4DrivFPnjdCgv7nTH1gkhNaHfvo0xPxKdKXACCbDdVZ0y578X3cv6OXLs95yqmcP9fe1aTZbMrGuOy2U3UhcN0xGLENf9echnzFVfEPT8aEpCUu2wV4elvnQB41UQDakN856FlZHqy6I7FLTm2PHh39Ts6YixZ3dajnSXRiJzpwc9D99mT2rB0tXc1HrAXbe0Rrn60udt+fO/R5q4L+Z3nT+parnhTMQteg3+2dgrwAuu5x4zlyeUf+R5rqWmpPaoSAS6XYywai3PFg6917fMBIwb3GAesLuTdBO7oSP9Q6JFjh3Hz2Yf3KAnHVHPqTdbflKMkMgX4GvCGiCxzaf+KFzzmiMglwBrgPLfsKbzuvS14XXy/AaCqm0XkFuBVt97Nqrq5FDuwoyPWrVdVocQVfnj2RJ59cwOL3tqUcwABsvZeydeuIjU0F1M8rnQqRKK5PRAaRDQO0XjiAt7zWZX7X3qfea+tdYEsnjGAJOT6S3fGlMVvbuCOxS09liUag5vXbmXK+JG8NPOUbtVT3Z+z8LY4Y95yfnruET0CbWdMWbl+GycdsjfTJo1hwn57csYvX6QjGifRTBLNNdcivNG6LeebruRuv34BLpOYdp/3ZMGVJ/SomeqIw4r125g6cd+Mw6VcNPkAULrmCkpMpGWlkJ5KHkRU9UXSVTrCqT7rK3BFmu+6H7i/cLnLzYp12woeQBKuf2wFQa7fFdKcUl45/m61rpU8FtOC/m47IjF+/zevuqgYgSwscLdPj7hBtSHa2juYctti37aFTJOXrWn71Hdbn+yM0rx2K4Prwixbu9W7aUrapQG1IeJxb6zqdCWS2rA36+EvfYJeuv1LvlCvXL8tYwAJh2BQbQ2RaAwR6VZaqQ2FeGrFR76l81sWeEPZZJvi4KLjDsh70MhKU4rhW2zsrDxlevCpEKqwANArNSH4j69MYvuuKNfl2ZCeKtenbBS44n99xveOvpLFFAaFevaci8Ti3LX4XTpi2m0U32GDahk9dCDL1m6lJtSz5BzpjHPHond7bEeA787xKgk6Ypp2BszfXzqZC+/9G37R++vHH8BDr35AJAbRHHvahUPSNUDp7pz4qw0LT3/7xK7qvVuffqvb8o5YnLufe9e3s0qi2ipRyko8fDl8cB3Na7d2u+BW8+RUpRq+xYJIHtraIzz31sa8+ribzKJxOGzfPRk+uI5Fb27I2nhakG3GtOoCSILfTYbGoSPlQh6JKpc+0ERHTKkNgd+ILqraIzjUhrxglRxwktcZXB8m5kZirq0J4xdABtSEOHzMUOpeC6ctkfk1XNenNFyv3exfSgK45rRDGb/PHrS1R/jKsz1HkYhE42m7ZSca4B98ZU3XeGafdkQREerC3r7P+MJhnHNMAyvXbwMk6/wylaaU86xYEMlRIqqr5tYl1ORu7mut3PfiewVtBO+rhtWH+US8Z3k+dd1q093nJ47T5AAyqNYb6faE8SN59f3N7EiKEANrQ3xzygHc/bx/p4pBtSEumXIgJx08ktqaMAua1/ueC9F4nOGDajOODu3boSOlPSRTiX+vwV6X5ZXrt5GuoOO3jZB4D6X+84NLuzondDWuq3aVXG59+q1upZtE1/NqGYixlPOs2ACMOchltFcTXD690PqK3nbMuPFLE7jxifyrVSfsO4RVH7UH3m6it1NyL8JU+XZVT4xCff2ZEzh89NCuIfUzjZr97L+cxMoPP+GaR5tLdlNXXyP89dpTq6JEUoyBJCtmAMZqlMu8Eya4/hZAph25H39auYHe7PmilR9mX8lHbwII7G5Ez/Swbb61vdM/dwDDB9Vx0/zdPaGu/+KEtKMWJGqpZs5bXtJagbBUz4yJmbp6F5oFkRwEHYbDlIcA3z/jMG7701u+bQHl9kTzh9SGezfi0N/WFG2c0ZL7zz+v7ro9S4wwcOP8FVzzhcP46cJ3fIbAD/Niy8clv7GLaXXNmJjr5GK9Vdaxs6pFIqrnM16WKR8FDtl3Dy44dv9yZ8WX4jXu9kaFDPhcEH6/RGccfvynt5hy0Igeyz7tjHHrU2+yK1q6HyF5Sulq9N6mduY2rU074GZvWEkkR1PGj8xpoD5TGdZt2dnrITtMecUVnnvHv7deqaqxQuJN71MbDnHzglXsMaCmahrXE52BOqPxbp0MMk0BEISVRHLUumUndYUeytUUzUstH+c8nIjpHxoPSD8fSLrZHeLqlRx3ReNEovGukYcTso1AXC7JXXxT422mKQCCsJJIjhqGDyz4eFmmeHIZgdj0L01r/Ef1hdwf8u2MKS//vY0zjxxd0XOxt27ZmbH6fdnarVlntsyVlUTycMkJB5Y7C8aYMrvq4deZ9cLfmTG3udtcJdfMba6YEsmKddsyzpQ6qYCzNFoQyUFiprjZL79f7qwYY8ospnDrU2/1mA0zElV+v+SDNJ8qnbb2CDcv6DlWWsIXD9+3YKUQsCCSVXLd4g6/AYSMMca567mWspdGWrfsJCzpL+3/MGHvtMuCsCCSRdD5vo0x/U9d2HsgsZwahg/sNmFZqnlLM88zny+7OmbRMHwgOzqKNz+FMabvSB7/q1xGDKnnvMb0z0j19hmlVBZEstiyo6PkU9YaY6pPTYiKmbjqG1PGpV225dOOgm7LgkgWy9am7xZojDEJ4VAoZT6U8tmaIVC8s3FHQZ8TsSCSRSG7whlj+i5VLXt7SMKMecszLv/j64UbzcGCSBZr2naUOwvGmCrQEVMG14XLnQ1aNmxn9cfpJ/QCeHdD70ZzTmZBJItZL6wudxaMMVXi6rnLyp2FnEoZG7cXrsRU9UFERKaKyNsi0iIi1xb6+99vK1zENsb0bcvWfkLTe21lzcPrH2SfJmBFq7WJACAiYeBu4HRgAnCBiEwo5DY2bu8s5NcZY/q4uxa3lHX7r+Uw10whH1qo6iACHAu0qOpqVe0AHgbOKuQGrHevMSYfr77/cVm3v6vEc81UexAZA6xNet/q0roRkctFpElEmjZt8p+fwBhjCqGjzBOGDa7Nvs6QusJtr9qDSE5UdZaqNqpq46hRo/L67JcOL+w4M8aYvm1qgcemytcP//GIrOv89hvHFWx71R5E1gFjk943uLSC+eVXP1vIrzPG9HHlvmacffRY9tszfVHjxPEjaDyw57TDQUk1T/kqIjXAO8CpeMHjVeBCVU07DnJjY6M2NTXlva1v/e5VnlixMWhWjTF5EsrbJjl22ADaduzi004YEIb6mjAqMGnMMA4bvScfbP6Uw/bdg5dbNvHG+k849dBRZQ8gyR57bS2/W7KGrTuibI908JlRQ/juPxwaOICIyFJVbeyRXs1BBEBEzgD+AwgD96vqDzOtHzSIGGNMf5YuiFT99Liq+hTwVLnzYYwx/VG1t4kYY4wpIwsixhhjArMgYowxJjALIsYYYwKr+t5Z+RKRTcCagB8fCZR3TIP8WH6Lp5ryCpbfYqum/AbN6wGq2uNp7X4XRHpDRJr8urhVKstv8VRTXsHyW2zVlN9C59Wqs4wxxgRmQcQYY0xgFkTyM6vcGciT5bd4qimvYPkttmrKb0Hzam0ixhhjArOSiDHGmMAsiBhjjAnMgkgORGSqiLwtIi0icm258wMgIveLyEYRWZGUtpeILBSRd92/w126iMidLv/LReToMuR3rIg8JyKrRGSliFxVyXkWkQEi8jcRaXb5vcmlHygiS1y+HhGROpde7963uOXjSplfl4ewiLwuIguqIK/vi8gbIrJMRJpcWkUeCy4Pw0Rkroi8JSJvisjxlZpfETnU/a6Jv09E5DtFy6+q2l+GP7wh5v8OHATUAc3AhArI10nA0cCKpLTbgWvd62uB29zrM4Cn8aZoOA5YUob87gcc7V7vgTcPzIRKzbPb7hD3uhZY4vIxBzjfpf8a+L/u9T8Dv3avzwceKcNv/F3g98AC976S8/o+MDIlrSKPBZeHB4BL3es6YFgl5zcp32HgI+CAYuW3LDtWTX/A8cAzSe+vA64rd75cXsalBJG3gf3c6/2At93r/wQu8FuvjHl/HPiHasgzMAh4DZiM96RvTeqxATwDHO9e17j1pIR5bAAWAacAC9wFoSLz6rbrF0Qq8lgAhgLvpf5GlZrflDyeBrxUzPxadVZ2Y4C1Se9bXVol2kdVP3SvPwL2ca8rah9c9clReHf3FZtnVz20DNgILMQrkW5V1ahPnrry65ZvAwo3B2l2/wHMAOLu/QgqN6/gTVr4/0RkqYhc7tIq9Vg4ENgE/JerLrxXRAZTuflNdj7wkHtdlPxaEOmj1LulqLj+2yIyBJgHfEdVP0leVml5VtWYqk7Cu8s/FjiszFnyJSJnAhtVdWm585KHE1T1aOB04AoROSl5YYUdCzV4Vcf3qOpRwA686qAuFZZfAFwb2DTg0dRlhcyvBZHs1gFjk943uLRKtEFE9gNw/yYmha+IfRCRWrwA8qCq/sElV3SeAVR1K/AcXpXQMBFJzAianKeu/LrlQ4G2EmVxCjBNRN4HHsar0rqjQvMKgKquc/9uBP6IF6Qr9VhoBVpVdYl7PxcvqFRqfhNOB15T1Q3ufVHya0Eku1eBg11Plzq84uH8MucpnfnAdPd6Ol67QyL9YtcL4zhgW1KxtiRERID7gDdV9edJiyoyzyIySkSGudcD8dpv3sQLJuemyW9iP84FFru7vaJT1etUtUFVx+Edn4tV9aJKzCuAiAwWkT0Sr/Hq7VdQoceCqn4ErBWRQ13SqcCqSs1vkgvYXZWVyFfh81uOxp5q+8PrvfAOXp3498udH5enh4APgU68O6VL8Oq1FwHvAs8Ce7l1Bbjb5f8NoLEM+T0Br/i8HFjm/s6o1DwDRwCvu/yuAP7NpR8E/A1owasmqHfpA9z7Frf8oDIdF59nd++sisyry1ez+1uZOKcq9VhweZgENLnj4TFgeIXndzBe6XJoUlpR8mvDnhhjjAnMqrOMMcYEZkHEGGNMYBZEjDHGBGZBxBhjTGAWRIwxxgRWk30VY4wfEYnhdYmswRtb6WvqPZhoTL9hJRFjgtupqpNU9XBgM3BFuTNkTKlZEDGmMF7GDVonIp8RkT+5wQX/IiKHufR9ROSP4s1R0iwin3PpXxVv7pJlIvKfIhJ26e0i8kO37isisk+67xGRm0XkO4nMuM9dVfJfwfQ7FkSM6SV30T+V3cPhzAK+parHAFcDv3LpdwJ/VtUj8cZeWiki/wP4CjBFvcEeY8BFbv3BwCtu/ReAy9J9D3A/cLHLTwhv+JPfFWePjdnN2kSMCW6gGyp+DN64WgvdKMWfAx71hgsDoN79ewruQq+qMWCbiHwNOAZ41a0/kN0D43XgzQ0CsBRv/C7f73Hf1SYiR+EN8f26qpZ0UEXTP1kQMSa4nao6SUQG4U30dAXwW7x5PCbl+B0CPKCq1/ks69Td4xLFyH6+3gt8HdgXr2RiTNFZdZYxvaSqnwLfBr4HfAq8JyJfhq75q490qy4C/q9LD4vIUJd2rojs7dL3EpEDsmzS73vAG1J9KvBZvKBmTNFZEDGmAFQ1MeLvBXhtGpeISGKU2rPcalcBJ4vIG3jVUxNUdRXwA7xZ/pbjzaC4X5bN9fgel4cOvOHf57hqLmOKzkbxNaaPcA3qrwFfVtV3y50f0z9YScSYPkBEJuDND7LIAogpJSuJGGOMCcxKIsYYYwKzIGKMMSYwCyLGGGMCsyBijDEmMAsixhhjAvv/z+p2J/4sw28AAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "best_customer = rfm_segmentation.loc[rfm_segmentation['RFMScore']=='444']\n",
        "corr_matrix = best_customer.corr()\n",
        "sns.heatmap(corr_matrix)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 330
        },
        "id": "T3TrARuE51bH",
        "outputId": "0641e7e5-21a0-44b0-f9db-dce940f155bb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fd81e8edf10>"
            ]
          },
          "metadata": {},
          "execution_count": 77
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "best_cust_list = list(best_customer.index)\n",
        "loyal_cust_list = list(loyal_customer.index)\n",
        "big_spenders_list = list(big_spenders.index)\n",
        "almost_lost_list = list(almost_lost.index)\n",
        "lost_cust_list = list(almost_lost.index)\n",
        "lost_cheap_list = list(almost_lost.index)"
      ],
      "metadata": {
        "id": "POjGtptb6ssz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(rfm_segmentation.index)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NNhM0UmB7X1k",
        "outputId": "881861ca-d557-4bfd-96e3-5bd874cc9bcd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "RangeIndex(start=0, stop=93341, step=1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rfm_segmentation['Segment'] = ['Others']*93341       \n",
        "rfm_segmentation"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "id": "0tJb1PgBQaha",
        "outputId": "c858714e-30f3-45c9-9033-f55017590088"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                         Cust_unique_Id  Recency  Frequency  Monetary  \\\n",
              "0      0000366f3b9a7992bf8c76cfdf3221e2      111          1    141.90   \n",
              "1      0000b849f77a49e4a4ce2b2a4ca5be3f      114          1     27.19   \n",
              "2      0000f46a3911fa3c0805444483337064      537          1     86.22   \n",
              "3      0000f6ccb0745a6a4b88665a16c9f078      321          1     43.62   \n",
              "4      0004aac84e0df4da2b147fca70cf8255      288          1    196.89   \n",
              "...                                 ...      ...        ...       ...   \n",
              "93336  fffcf5a5ff07b0908bd4e2dbc735a684      447          1   2067.42   \n",
              "93337  fffea47cd6d3cc0a88bd621562a9d061      262          1     84.58   \n",
              "93338  ffff371b4d645b6ecea244b27531430a      568          1    112.46   \n",
              "93339  ffff5962728ec6157033ef9805bacc48      119          1    133.69   \n",
              "93340  ffffd2657e2aad2907e67c3e9daecbeb      484          1     71.56   \n",
              "\n",
              "       R_Quartile  F_Quartile  M_Quartile RFMScore Segment  \n",
              "0               4           1           3      413  Others  \n",
              "1               4           1           1      411  Others  \n",
              "2               1           1           2      112  Others  \n",
              "3               2           1           1      211  Others  \n",
              "4               2           1           4      214  Others  \n",
              "...           ...         ...         ...      ...     ...  \n",
              "93336           1           1           4      114  Others  \n",
              "93337           2           1           2      212  Others  \n",
              "93338           1           1           3      113  Others  \n",
              "93339           3           1           3      313  Others  \n",
              "93340           1           1           2      112  Others  \n",
              "\n",
              "[93341 rows x 9 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0f3ea652-a5db-4ed7-8001-0f90c76c37bd\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Cust_unique_Id</th>\n",
              "      <th>Recency</th>\n",
              "      <th>Frequency</th>\n",
              "      <th>Monetary</th>\n",
              "      <th>R_Quartile</th>\n",
              "      <th>F_Quartile</th>\n",
              "      <th>M_Quartile</th>\n",
              "      <th>RFMScore</th>\n",
              "      <th>Segment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>\n",
              "      <td>111</td>\n",
              "      <td>1</td>\n",
              "      <td>141.90</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>413</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>\n",
              "      <td>114</td>\n",
              "      <td>1</td>\n",
              "      <td>27.19</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>411</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0000f46a3911fa3c0805444483337064</td>\n",
              "      <td>537</td>\n",
              "      <td>1</td>\n",
              "      <td>86.22</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>112</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0000f6ccb0745a6a4b88665a16c9f078</td>\n",
              "      <td>321</td>\n",
              "      <td>1</td>\n",
              "      <td>43.62</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>211</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0004aac84e0df4da2b147fca70cf8255</td>\n",
              "      <td>288</td>\n",
              "      <td>1</td>\n",
              "      <td>196.89</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>214</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93336</th>\n",
              "      <td>fffcf5a5ff07b0908bd4e2dbc735a684</td>\n",
              "      <td>447</td>\n",
              "      <td>1</td>\n",
              "      <td>2067.42</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>114</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93337</th>\n",
              "      <td>fffea47cd6d3cc0a88bd621562a9d061</td>\n",
              "      <td>262</td>\n",
              "      <td>1</td>\n",
              "      <td>84.58</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>212</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93338</th>\n",
              "      <td>ffff371b4d645b6ecea244b27531430a</td>\n",
              "      <td>568</td>\n",
              "      <td>1</td>\n",
              "      <td>112.46</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>113</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93339</th>\n",
              "      <td>ffff5962728ec6157033ef9805bacc48</td>\n",
              "      <td>119</td>\n",
              "      <td>1</td>\n",
              "      <td>133.69</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>313</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>93340</th>\n",
              "      <td>ffffd2657e2aad2907e67c3e9daecbeb</td>\n",
              "      <td>484</td>\n",
              "      <td>1</td>\n",
              "      <td>71.56</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>112</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>93341 rows × 9 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0f3ea652-a5db-4ed7-8001-0f90c76c37bd')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-0f3ea652-a5db-4ed7-8001-0f90c76c37bd button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-0f3ea652-a5db-4ed7-8001-0f90c76c37bd');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for i in rfm_segmentation.index:\n",
        "    if i in lost_cheap_list:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Lost Cheap Customers'\n",
        "    elif i in lost_cust_list:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Lost Customer'\n",
        "    elif i in best_cust_list:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Best Customers'\n",
        "    elif i in almost_lost_list:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Almost Lost'        \n",
        "        \n",
        "#rfm_segmentation.head(40)\n",
        "#cust_id = rfm_segmentation['Cust_unique_Id'].tolist()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V3qEGIUSO0sD",
        "outputId": "85e2aa8b-e0cc-4dfc-d8b2-b67c5b113885"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  self._setitem_single_block(indexer, value, name)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "loyal2 = []\n",
        "for i in loyal_cust_list:\n",
        "    if i not in best_cust_list and i not in lost_cheap_list and i not in almost_lost_list and i not in lost_cust_list:\n",
        "         loyal2.append(i)\n",
        "for i in rfm_segmentation.index:\n",
        "    if i in loyal2:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Loyal Customers'\n",
        "\n",
        "big_spenders2= []\n",
        "for i in big_spenders_list:\n",
        "     if i not in best_cust_list and i not in lost_cheap_list and i not in almost_lost_list and i not in lost_cust_list:\n",
        "         big_spenders2.append(i)\n",
        "\n",
        "for i in rfm_segmentation.index:\n",
        "    if i in big_spenders2:\n",
        "        rfm_segmentation.Segment.iloc[i]= 'Big Spenders'\n",
        "rfm_segmentation.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        },
        "id": "MXGoUHFfQnOz",
        "outputId": "89b005e0-d8bc-4c22-d0d9-3ca695078493"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     Cust_unique_Id  Recency  Frequency  Monetary  R_Quartile  \\\n",
              "0  0000366f3b9a7992bf8c76cfdf3221e2      111          1    141.90           4   \n",
              "1  0000b849f77a49e4a4ce2b2a4ca5be3f      114          1     27.19           4   \n",
              "2  0000f46a3911fa3c0805444483337064      537          1     86.22           1   \n",
              "3  0000f6ccb0745a6a4b88665a16c9f078      321          1     43.62           2   \n",
              "4  0004aac84e0df4da2b147fca70cf8255      288          1    196.89           2   \n",
              "\n",
              "   F_Quartile  M_Quartile RFMScore       Segment  \n",
              "0           1           3      413        Others  \n",
              "1           1           1      411        Others  \n",
              "2           1           2      112        Others  \n",
              "3           1           1      211        Others  \n",
              "4           1           4      214  Big Spenders  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-743be907-e34e-411c-8d40-82fe8673d9b1\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Cust_unique_Id</th>\n",
              "      <th>Recency</th>\n",
              "      <th>Frequency</th>\n",
              "      <th>Monetary</th>\n",
              "      <th>R_Quartile</th>\n",
              "      <th>F_Quartile</th>\n",
              "      <th>M_Quartile</th>\n",
              "      <th>RFMScore</th>\n",
              "      <th>Segment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>\n",
              "      <td>111</td>\n",
              "      <td>1</td>\n",
              "      <td>141.90</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>413</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>\n",
              "      <td>114</td>\n",
              "      <td>1</td>\n",
              "      <td>27.19</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>411</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0000f46a3911fa3c0805444483337064</td>\n",
              "      <td>537</td>\n",
              "      <td>1</td>\n",
              "      <td>86.22</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>112</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0000f6ccb0745a6a4b88665a16c9f078</td>\n",
              "      <td>321</td>\n",
              "      <td>1</td>\n",
              "      <td>43.62</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>211</td>\n",
              "      <td>Others</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0004aac84e0df4da2b147fca70cf8255</td>\n",
              "      <td>288</td>\n",
              "      <td>1</td>\n",
              "      <td>196.89</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>214</td>\n",
              "      <td>Big Spenders</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-743be907-e34e-411c-8d40-82fe8673d9b1')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-743be907-e34e-411c-8d40-82fe8673d9b1 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-743be907-e34e-411c-8d40-82fe8673d9b1');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Customer Segmentation"
      ],
      "metadata": {
        "id": "yN3x-0irRrvW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sq1 = rfm_segmentation.groupby('Segment')['Cust_unique_Id'].nunique().sort_values(ascending=False).reset_index()\n",
        "plt.figure(figsize=(10,5))\n",
        "plt.title('Customer Segmentation', weight = 'bold')\n",
        "sq1.drop([0], inplace= True)\n",
        "sns.barplot(data=sq1, x='Segment', y='Cust_unique_Id', palette=\"deep\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 367
        },
        "id": "KJXbn_xVRuQj",
        "outputId": "088989ed-f766-4343-d3c1-cd7d669f6e06"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fd81d6786d0>"
            ]
          },
          "metadata": {},
          "execution_count": 83
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Best Seller Distribution"
      ],
      "metadata": {
        "id": "eCqIB4XxMmXC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#visualisasi best seller\n",
        "best_customer = seller_seg_2[seller_seg_2['SellerScore']=='4444']\n",
        "#calculate and show correlations\n",
        "corr_matrix = best_customer.corr()\n",
        "sns.heatmap(corr_matrix)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 385
        },
        "id": "mOHOp4DSVFHp",
        "outputId": "4c3bc397-ac2b-4e70-d6b5-8e3aded8fa4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fd81ab82c50>"
            ]
          },
          "metadata": {},
          "execution_count": 84
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcgAAAFfCAYAAADKwlwRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd5xcVf3/8dc7oYTeRZoEMYqAEBKK0psKForUCEgwEhFEkR8oioWiqOD3i4hSokIACxAQjciXgJRQFMhCGiBNQGmKFCGACMm+f3+cM2QymZ2dzc7cuZt8njzuY2duO5/ZDfO559xzz5FtQgghhDCvQZ0OIIQQQiijSJAhhBBCHZEgQwghhDoiQYYQQgh1RIIMIYQQ6ogEGUIIIdQRCTKEEELpSbpA0rOS7u1huyT9SNIjkmZIGtHfMiNBhhBCGAjGA7s12L47MCwvY4Fz+1tgJMgQQgilZ/sW4IUGu+wJXOzkDmBFSWv0p8zF+nNwGFjefO7Rjg+btNSa23U6BADuHbppp0Pg4FmzOh0CAF3P3NrpEEI5qb8n6Mt3zhKrrf9ZUs2vYpztcX0obi3giar3T+Z1z/ThHPOIBBlCCKHjcjLsS0Jsu0iQIYQQ2qN7TpGlPQWsU/V+7bxugcU9yBBCCO3h7uaX/psIfCr3Zn0/8JLtBW5ehahBhhBCaBPPmd2yc0n6NbAjsKqkJ4FvAYsD2D4PuAb4CPAI8BpwWH/LjAQZQgihPbpbUjMEwPaoXrYbOKplBRIJMoQQQru0pum0YyJBhhBCaI9iO+m0XCTIEEII7RE1yBBCCGF+reyk0wmRIBuQtApwQ377dmAO8K/8fkvbb1Ttewxp5IfXWlT2ScArtn/QivOFEELhWthJpxMiQTZg+3lgODSVsI4BfkHqXhxCCGGAN7HGQAF9JGkXSVMlzczTrywp6QvAmsBNkm7q4bjBksZLujcf+6W8/nBJUyRNl3SlpKXrHLu+pGsl3S3pVkkb5PX75fNNl3RLOz93CCH0Wfec5pcSigTZN0NIU64cYPt9pBr452z/CHga2Mn2Tj0cOxxYy/bG+dgL8/rf2N7C9qbAX4AxdY4dBxxteyRwHHBOXv9N4MP52D36//FCCKGFih1Jp+UiQfbNYOAx2w/l9xcB2zd57KPAOyWdLWk34OW8fuNcK5wJHARsVH2QpGWBrYEJkqYB5wOVKVxuB8ZLOjzHNh9JYyV1Ser62cW/bjLUEEJoge7u5pcSinuQBbH9oqRNgQ8DRwD7A58m1Uj3sj1d0mjSUErVBgH/tj28zjmPkLQV8FHgbkkj833T6n3eGiG/DNNdhRAWIQO8F2vUIPtmDjBU0rvy+0OAyfn1LGC5ng6UtCowyPaVwNeBEXnTcsAzkhYn1SDnYftl4DFJ++XzKCdaJK1v+07b3yT1rl2n9vgQQugUe07TSxlFDbJvXicNgDtB0mLAFOC8vG0ccK2kp3u4D7kWcKGkykXJV/PPbwB3khLcndRPsgcB50r6Omlw3kuB6cAZkoaRJja9Ia8LIYRyKOm9xWYpje8aFgVlaGJdas3tOh0CAPcO3bTTIXDwrFmdDgGArmdu7XQIoZzU3xO8fs/Epr9zhozYo9/ltVrUIEMIIbTHAK9BRoJsA0l3AkvWrD7E9sxOxBNCCB1R0ucbmxUJsg1sb9XpGEIIoeNaO2HybsBZpEfafmb7ezXb30F69G7FvM8Jtq/pT5nRizWEEEJ7tGigAEmDgZ8AuwMbAqMkbViz29eBy21vBhzI3AFVFljUIEMIIbRH6wYA2BJ4xPajAJIuBfYE7q/ax8Dy+fUKpNHN+iUSZAghhPZoXYJcC3ii6v2TQO2trJOA6yQdDSwD7NrfQqOJNYQQQlv0ZaCA6mEx8zK2j8WNAsbbXhv4CHBJ1XPnCyRqkCGEENqjD510qofFrOMp5h0pbO28rtoYYLd8rj9LGgKsCjzbdBA1IkEuQsrwkP5/ni7HQ+l7bHZUp0Ng0ubRgBMWcq1rYp0CDJO0HikxHgh8smafvwO7kCZweC9p9qV/0Q+RIEMIIbRHiwYKsD1b0ueBSaRHOC6wfZ+kU4Au2xOB/wf8NM+1a2C0+zlUXCTIEEII7dHCaazyM43X1Kz7ZtXr+4FtWlYgkSBDCCG0Sww1F0IIIdRR0omQmxUJMoQQQnsM8AmTI0GGEEJoj6hBhhBCCHXEPcgQQgihjqhBhhBCCHVEDXJgkLQKcEN++3ZgDnNHWdjS9htV+x4DjLP9WrFR9p2kHYE3bP+p07GEEMI8ogY5MNh+HhgOIOkk4BXbP+hh92OAXwClT5DAjsArQCTIEEK5zJnT6Qj6ZZEeDFLSLpKmSpop6QJJS0r6ArAmcJOkmxoce24ecf4+SSdXrX9c0nclTcvbR0iaJOmvko7I+0jSGZLuzWUfkNfvKOnqqnP9WNLoqvOeLOmefMwGkoYCRwBfyuV1frDVEEKo6O5ufimhRaYGWccQYDywi+2HJF0MfM72DyUdC+xk+7kGx59o+4U80/UNkjaxPSNv+7vt4ZLOzGVsk8u7FzgP+ASpNrspabT5KZJuaSLm52yPkHQkcJztz0g6j8a14RBC6IySJr5mLco1yMHAY7Yfyu8vArbvw/H7S7oHmApsBGxYtW1i/jkTuNP2LNv/Av4raUVgW+DXtufY/icwGdiiiTJ/k3/eDQxtJsjqOda6u19t5pAQQmgNdze/lNCiXINcYHnKleOALWy/KGk8qYZY8d/8s7vqdeV9o9/5bOa9aBlSs71yrjm9nOct1XOsLbbEWv0a2T6EEPokapAD1hxgqKR35feHkGpyALOA5RocuzzwKvCSpNWB3ftY9q3AAZIGS1qNVHO9C/gbsGG+F7oiaW6z3vQWawghdMacOc0vJbQo1yBfBw4DJkhajDQh53l52zjgWklP296p9kDb0yVNBR4AngBu72PZVwEfAKaT5i37su1/AEi6nHSv8jFS821vfg9cIWlP4Gjb5ZiROIQQBngNUv2cTzIMIGVoYv3P0+XI33tsdlSnQ+AXw8vxFNEqv5/c+05hUaT+nuA/Pzu26e+cpT7zv/0ur9UW5SbWEEIIbeRuN730RtJukh6U9IikE3rYZ39J9+fH737V3/gX5SbWpki6E1iyZvUhtmd2Ip4QQhgwWtTEmh+n+wnwQeBJ0qNxE23fX7XPMOCrwDa58+Tb+ltuJMhe2N6q0zGEEMKA1LrHN7YEHrH9KICkS4E9gfur9jkc+IntFwFsP9vfQqOJNYQQQnvMntP0Uv3Mdl7GVp1pLVKHyIon87pq7wbeLel2SXdI2q2/4UcNMoQQQnv0oYm1+pntBbQYMIw0PvXawC2S3mf73wt6wqhBhhBCaA+7+aWxp4B1qt6vnddVexKYaPtN248BD5ES5gKLBBlCCKE9WjdY+RRgmKT1JC0BHMjcIT0rfkuqPSJpVVKT66P9CT+aWEMIIbRHE49vNMP2bEmfByaRxtG+wPZ9kk4BumxPzNs+JOl+0khpx+dpDhdYJMhFyL1DN+10CKV4QB9g4tSfdDoE3jjzK50OIYT2auEQcravAa6pWffNqtcGjs1LS0SCDCGE0BYe4EPNRYIMIYTQHi1qYu2USJAhhBDao6TzPDYrEmQIIYT2iBpkCCGEUEfcgwwhhBDqKOlEyM2KBBlCCKE9ook1hBBCmF885hFCCCHUEzXIEEIIoY4BniAX2cHKJe0lyZI26HQsFZJOkbRrp+MIIYSWcHfzSwktyjXIUcBt+ee3Wn1ySYvZnt2XY6rHFQwhhIHOs8uZ+Jq1SNYgJS0LbAuMAQ6UtJukCVXbd5R0dX49RtJDku6S9FNJP25w3vGSzpN0J3C6pPUlXSvpbkm3StpA0gqS/iZpUD5mGUlPSFo8H79vXj9S0uR87CRJa0h6m6S78/ZNcw34Hfn9XyUt3a7fWQgh9Fm3m19KaJFMkMCewLW2HwKeB14EtpK0TN5+AHCppDWBbwDvB7YBmmmOXRvY2vaxpNmxj7Y9EjgOOMf2S8A0YIe8/8eASbbfrJxA0uLA2cC++dgLgO/YfhYYIml5YDugC9hO0rrAs7Zfqw1G0lhJXZK6Ln/p703/gkIIod9aNx9kRyyqTayjgLPy60uB/YBrgY9LugL4KPBlYBdgsu0XAHIt8929nHuC7Tm5lro1MEFSZduS+edlpCR8E2niz3NqzvEeYGPg+nzsYOCZvO1PpGS9PXAasBsg4NZ6wdgeR0rUPPDuj5TzMi2EsHAqac2wWYtcgpS0MrAz8D5JJiUfA4cBRwEvkCbgnFWV2Pri1fxzEPBv28Pr7DMROC3HMhK4sTZM4D7bH6hz7C2k2uO6wO+Ar+T4/7AgwYYQQtu0MEFK2o1UsRkM/Mz293rYbx/gCmAL2139KXNRbGLdF7jE9rq2h9peB3gMmA2MAA4n1SoBpgA7SFpJ0mLAPs0WYvtl4DFJ+wEo2TRveyWf+yzgatu14zE9CKwm6QP52MUlbZS33QocDDxsu5uU0D9C6nAUQgil4TndTS+NSBoM/ATYHdgQGCVpwzr7LQd8EbizFfEviglyFHBVzborSU2dV5P+AFcD2H6K1Ix5F3A78DjwUh/KOggYI2k6cB/p3mfFZaREd1ntQbbfICXy7+djp5Gaa7H9OKmGeUve/TZSTfXFPsQVQgjt17pOOlsCj9h+NH8/Xsq836cVpwLfB15vRfiLXBOr7Z3qrPtR1dvP12z+le1xuQZ5FfDbBuceXfP+MdI9wnr7XkFKdHWPtz2NdJ+x3rHrVL0+jZTEQwihVNyHJlZJY4GxVavG5T4UAGsBT1RtexLYqub4EcA6tv8g6fgFi3hei1yCXAAn5Yf3hwDX0SBBhhBCqNKHBFndobCv8mNz/wuMXpDjexIJshe2j6tdJ+lEUs/XahNsf6eYqEIIYQBo3dMbTwHrVL1fO6+rWI7U8//m3Lny7cBESXv0p6NOJMgFkBNhJMMQQmigL02svZgCDJO0HikxHgh88q1y0vPlq1beS7oZOK6/vVgjQYYQQmiP2a1JkLZnS/o8MIn0mMcFtu+TdArpsbyJLSmoRiTIEEIIbdHCGiS2rwGuqVlXd/xq2zu2osxIkCGEENqjnCPINS0SZAghhLZoZQ2yEyJBhhBCaI+oQYaB4uBZszodApM2L8fgTW+c+ZVOh8ASX/p+p0MIoa1KOg9y0yJBhhBCaIu+TRlfPpEgQwghtEfUIEMIIYT5RRNrCCGEUEckyBBCCKGOSJAhhBBCHZ6j3ncqsUiQIYQQ2sLdkSBDCCGE+UQTawghhFCHPbBrkOUY1qQNJO0lyZI2aGMZrzS7j6Q1JV3RrlhCCKFs3N38UkYLbYIERgG35Z8dZ/tp2/v29zySotYfQhgQ3K2mlzJaKBOkpGWBbYExwIGSdpM0oWr7jpKuzq/HSHpI0l2Sfirpxw3Ou56kP0uaKenbNduOlzRF0gxJJ9c5dqike/PrOyRtVLXtZkmbS1pG0gU5lqmS9szbR0uaKOlG4AZJF0vaq+r4X1b2DSGEsuieo6aX3uTv8QclPSLphDrbj5V0f/4OvkHSuv2Nf6FMkMCewLW2HwKeB14EtpK0TN5+AHCppDWBbwDvB7YBemuOPQs41/b7gGcqKyV9CBgGbAkMB0ZK2r7BeS4D9s/HrgGsYbsLOBG40faWwE7AGVUxjwD2tb0D8HNgdD5+BWBr4A+9xB5CCIVqVQ1S0mDgJ8DuwIbAKEkb1uw2Fdjc9ibAFcDp/Y1/YU2Qo4BL8+tLgf2Aa4GP5ybKjwK/IyW0ybZfsP0mMKHeyapsA/w6v76kav2H8jIVuIeUaIc1OM/lQKW5dX/SH7NynhMkTQNuBoYA78jbrrf9AoDtycAwSavlz3qlXX9YYEljJXVJ6vrXa//o5eOFEELr2M0vvdgSeMT2o7bfIH2vz9NqZvsm26/lt3cAa/c3/oXufpaklYGdgfdJMjAYMHAYcBTwAtBle5a0QO3e9f6UAr5r+/ymTmA/Jel5SZuQarNHVJ1nH9sP1nymrYBXa05zMXAwcCDps/VU1jhgHMDma2w3sGcvDSEMKC28t7gW8ETV+yeBrRrsPwb4v/4WujDWIPcFLrG9ru2httcBHgNmk5opD2du7XIKsIOklXLNcp9ezn07KSEBHFS1fhLw6XzvE0lrSXpbL+e6DPgysILtGVXnOVo5c0varMHx44FjAGzf30tZIYRQOFtNL9WtXXkZuyBlSjoY2Bw4o7/xL3Q1SFKTY+1MtFeSEtvVpHt3h8JbNbnTgLtINcsHgJcanPuLwK8kfYXUREs+z3WS3gv8Oee2V0i1u2cbnOsK0j3NU6vWnQr8EJghaRApsX+s3sG2/ynpL8BvG5QRQggd05fHN6pbu+p4Clin6v3aed08JO1K6suxg+3/Nl96fXITjb8LM0nL2n4l1yCvAi6wfVWn4+qNpKWBmcAI242S+lvK0MQ6afNyNFoM2aS3Cn77LfGl2uu4zlh81Xd2OoRQTv1uH31wg92b/s55zwP/12N5+fv5IWAXUmKcAnzS9n1V+2xGqnjsZvvhBQ66Sjm+rTrrpNwp5l5Sja30NbJ8lfQX4Oxmk2MIIRStVb1YcyfEz5NuQ/0FuNz2fZJOkbRH3u0MYFlggqRpkib2N/6FsYm1T2wfV7tO0omknq/VJtj+TjFRNWb7j0C/n/EJIYR2amUDpe1rgGtq1n2z6vWurSstWeQTZD05EZYiGYYQwkBV1hFymhUJMoQQQlt0D/DByiNBhhBCaIvuqEGGEEII84saZAghhFDHQJ8PMhJkCCGEthjoj9lHggwhhNAW0cQaBoyuZ27tdAghhEVINLGGEEIIdcyJBBlCCCHML5pYQwghhDqiiTWEEEKoow+zXZVSJMgQQght4f7PmNVRkSBDCCG0xexoYg0hhBDmN9BrkDFhcgghhLbo7sPSG0m7SXpQ0iOSTqizfUlJl+Xtd0oa2t/4C0uQklbJszxPk/QPSU9VvV+iZt9jJC3dwrJPkjTfxMj9ON+2ku6S9ED+gx3ZqnPn83+t5v2f8s+hku5tZVkhhNAuRk0vjUgaDPwE2B3YEBglacOa3cYAL9p+F3Am8P3+xl9YgrT9vO3htocD5wFnVt7bfqNm92OAliXIVpL0duBXwBG2NwC2AcZI2rsF55akQcA8CdL21v09dwghFK2FNcgtgUdsP5rzxaXAnjX77AlclF9fAewiqV9tvB1tYpW0i6SpkmZKuiBXkb8ArAncJOmmHo4bLGm8pHvzsV/K6w+XNEXSdElX1quFSlpf0rWS7pZ0q6QN8vr98vmmS7qlQdhHAeNt3wNg+zngy8Dx+TzjJe1bVd4r+eeykm6QdE+Oec+8fmiuhV4M3Av8HFgq16x/WX2OOr+DM/LnnSHps738ukMIoVAtTJBrAU9UvX8yr6u7j+3ZwEvAKv0Iv6OddIYA44FdbD+UE8TnbP9Q0rHATjn51DMcWMv2xgCSVszrf2P7p3ndt0lV7rNrjh1Hqv09LGkr4BxgZ+CbwIdtP1V1vno2Yu5VSkUXqdrfyOvA3rZflrQqcIekiXnbMOBQ23fk2PfLNe1GxgAv2d5C0pLA7ZKus/1YL8eFEEIh5vShAidpLDC2atU42+NaHlQfdLIGORh4zPZD+f1FwPZNHvso8E5JZ0vaDXg5r9841wpnAgeRktlbJC0LbA1MkDQNOB9YI2++HRgv6fAcW6sJOE3SDOCPpKud1fO2v1WSYx98CPhU/hx3kq6Uhs1XqDRWUpekrnHjOvpvLYSwiOlGTS+2x9nevGqp/sJ6Clin6v3aeR319pG0GLAC8Hx/4h+Qj3nYflHSpsCHgSOA/YFPk2qke9meLmk0sGPNoYOAf9erndk+ItcoPwrcLWmk7Xq/3PuBkcDvqtaNJNUiAWbncsj3EysdkA4CVgNG2n5T0uOkWjTAq8198nkIONr2pEY75X9klX9oA3x2thDCQNLCL5wpwDBJ65ES4YHAJ2v2mQgcCvwZ2Be40e7fjJSdrEHOAYZKeld+fwgwOb+eBSzX04G5iXKQ7SuBrwMj8qblgGckLU5KSPOw/TLwmKT98nmUEy2S1rd9p+1vAv9i3quVaj8BRksano9bBfgOcGre/jgpYQLsASyeX68APJuT407Auj19PuDN/BkamQR8rrKfpHdLWqaXY0IIoTCtugeZ7yl+nvS99xfgctv3STpF0h55t58Dq0h6BDgWmO9RkL7qZA3ydeAwUnPnYqQrhPPytnHAtZKetr1TnWPXAi7MNTSAr+af3yA1N/4r/6yXZA8CzpX0dVLyuhSYDpwhaRipZnZDXjcf289IOhgYJ2kFYCgw2nYluf8U+J2k6cC1zK0d/hL4fW7+7QIe6PE3kz7/DEn32J4v0Wc/y2Xfk3tq/QvYq8E5QwihUN3960Q6D9vXANfUrPtm1evXgf1aViCgftZAF3n5GcjPAdvbfrHT8fQi/tghhGb1O7tNWOOgpr9z9nvml6UbdmdA3oMsE9vnkHrChhBCqDK7dCmvb0qfICXdCSxZs/oQ2zPbXO6HmX8khsds93tAgBBCWBR0D/CxWEufIG1v1aFyJ5FuCIcQQlgAA/2eTukTZAghhIGpe2BXICNBhhBCaI9mZukos0iQIYQQ2mJO1CBDCCGE+UUNMoQQQqgjEmQIIYRQh6OJNYQQQphf1CBDCCGEOiJBhhBCCHVEL9YQQgihjqhBhhBCCHVEggwhhBDqGOhjsQ7qfZcQQgih77rV/NIfklaWdL2kh/PPlersM1zSnyXdJ2mGpAN6O28kyBBCCG3R3Yeln04AbrA9DLghv6/1GvAp2xsBuwE/lLRio5NGggwhhNAWc3DTSz/tCVyUX18E7FW7g+2HbD+cXz8NPAus1uikC5QgJa0iaVpe/iHpqar3S9Tse4ykpReknKJJ2lHS1k3sN1bSA3npkrRjC2MYKumTVe83l/Sj/Hq0pB+3qqwQQminvtQg8/dqV9Uytg9FrW77mfz6H8DqjXaWtCWwBPDXRvstUCcd288Dw3NBJwGv2P5BD7sfA/yCVL0tux2BV4A/9bSDpI8BnwW2tf2cpBHARElb2X6qP4VLWgwYCnwS+BWA7S6gqz/nDSGETuhLvdD2OGBcT9sl/RF4e51NJ9acx5J6LFrSGsAlwKG2G7butqyJVdIukqZKminpAklLSvoCsCZwk6SbGhx7br5iuE/SyVXrH5f03Vwz7ZI0QtIkSX+VdETeR5LOkHRvLvuAvH5HSVdXnevHkkZXnfdkSffkYzaQNBQ4AvhSLm+7HsL9CnC87ecAbN8DXAgcVXXuVfPrzSXdnF9vmW8QT5X0J0nvyetHS5oo6UZS2/n3gO1yDF+q/RxVn2c1SVdKmpKXbXr5E4UQQqFaeQ/S9q62N66z/A74Z058lQT4bL1zSFoe+ANwou07eiuzVY95DAHGA7vYfkjSxcDnbP9Q0rHATpWE0oMTbb8gaTBwg6RNbM/I2/5ue7ikM3MZ2+Ty7gXOAz5Bqs1uCqwKTJF0SxMxP2d7hKQjgeNsf0bSeTSuDQNsBNxds64LOKyX8h4AtrM9W9KuwGnAPnnbCGCT/DvYMcfzMUiJvofznQWcafs2Se8AJgHv7SWGEEIoTH97p/bBROBQUgXjUOB3tTvk239XARfbvqKZk7aqBjkYeMz2Q/n9RcD2fTh+f0n3AFNJCWjDqm0T88+ZwJ22Z9n+F/Df3ANpW+DXtufY/icwGdiiiTJ/k3/eTWrWbLcVgAmS7gXOJH3Oiuttv9DH8+0K/FjSNNLvaHlJy9buVN2uP25cj60XIYTQcgV20vke8EFJD5O+G78Hb7Xi/Szvsz8pL42u6jMzvNFJOz5QgKT1gOOALWy/KGk8qYZY8d/8s7vqdeV9o/hnM+8FwJCa7ZVzzenlPLXuB0YCN1atG8nc+4TV5VaXeSpwk+29c3PuzVXbXu1D+RWDgPfbfr3RTjXt+gP9ud0QwgBS1Eg6uV/MLnXWdwGfya9/QeoP07RW1SDnAEMlvSu/P4RUkwOYBSzX4NjlSQniJUmrA7v3sexbgQMkDZa0GukK4S7gb8CG+V7oitT55dXRW6wApwPfl7QKpIdPgb2B8/P2x0kJE+Y2oUKqQVY68YzuZwwA1wFHV970diUUQghF68ZNL2XUqgT5Ouke3ARJM0kXDuflbeOAa3vqpGN7Oqlp9QFSz83b+1j2VcAMYDqpVvdl2/+w/QRwOele5eW5jN78Hti7UScd2xOBnwO3S3oEuA3YKzf7ApwMnCWpi3ThUHE68F1JU2lcY50BzJE0XdKXGuz3BWBzpREh7id1MAohhNJwH5Yykl3W0MovP5ZxIelC42CX/5dZ9vhCCOXR7y42xw0d1fR3zg8e/3XpJsfq+D3Igcz2bFJzcgghhBplbTptVqEJUtKdwJI1qw+xPbPIOJoh6URgv5rVE2x/pxPxhBDCQDOn911KrdAEaXurIsvrj5wIIxmGEMICctQgQwghhPnFhMkhhBBCHXEPMoQQQqhjYKfHSJAhhBDaJGqQIYQQQh0tGGO1oyJBhhBCaIvopBNCCCHUEY95hBBCCHVEDTKEEEKoo7v0w1M31qrZPEIIIYR5FDVhsqSVJV0v6eH8c6UG+y4v6UlJP+7tvJEgQwghtIX78F8/nQDcYHsYcEN+35NTgVuaOWkkyBBCCG3R3Yeln/YELsqvLwL2qreTpJHA6qQJ53sVCTKEEEJbdOOmF0ljJXVVLWP7UNTqtp/Jr/9BSoLzkDQI+B/guGZPGp10QgghtEVfmk5tjwPG9bRd0h+Bt9fZdGLNeSypXsFHAtfYflJqbm7mSJAhhBDaopWPedjetadtkv4paQ3bz0haA3i2zm4fALaTdCSwLLCEpFds93i/siVNrJL2kmRJG7TifK0g6RRJPf5C+3nusZIeyEuXpB1beO6hkj5Z9X5zST/Kr0c30/MqhBDKYI67m176aSJwaH59KPC72h1sH2T7HbaHkppZL26UHKF19yBHAbflny0nqc81XdvftP3HNsTyMeCzwLa2NwDGAr+QtFYLzr0YMBR4K0Ha7rL9hf6eO4QQilZgJ53vAR+U9DCwa35fqWD8bEFP2u8EKWlZYFtgDOGGX4oAACAASURBVHCgpN0kTajavqOkq/PrMZIeknSXpJ82qg1JGi/pPEl3AqdLWl/StZLulnSrpA0krSDpb/nmK5KWkfSEpMXz8fvm9SMlTc7HTpK0hqS3Sbo7b98014Dfkd//VdLSPYT2FeB4288B2L4HuBA4Kh/7uKRV8+vNJd2cX28p6c+Spkr6k6T35PWjJU2UdCOpe/L3SM0A0yR9qfr3V/P7WU3SlZKm5GWbZv5eIYRQlKIe87D9vO1dbA+zvavtF/L6LtufqbP/eNuf7+28rbgHuSdwre2HJD0PvAhsJWkZ268CBwCXSloT+AYwApgF3AhM7+XcawNb254j6QbgCNsPS9oKOMf2zpKmATsANwEfAybZfrNyE1bS4sDZwJ62/yXpAOA7tj8taYik5YHtgC5SYroNeNb2az3EtBFwd826LuCwXj7LA8B2tmfnpt/TgH3ythHAJrZfyM21x9n+WI5/xx7OdxZwpu3bcmKfBLy3lxhCCKEwMd1ValY9K7++FNgPuBb4uKQrgI8CXwZ2ASZXMnuuZb67l3NPyMlxWWBrYEJV76Ml88/LSEn4JuBA4Jyac7wH2Bi4Ph87GKh0B/4TsA2wPSlh7QYIuLXJz94XKwAXSRpGmkd08apt11d+L32wK7Bh1e9jeUnL2n6leqfcVXoswPnnn8/YsX3pOR1CCAvOA3youX4lSEkrAzsD78vdageTvvwPIzU5vgB02Z7VbLfaGq/mn4OAf9seXmeficBpOZaRpJrpPGEC99n+QJ1jbyHVHtcl3dT9So7/Dw1iur9OOSNJtUiA2cxtuh5Stc+pwE2295Y0FLi5atur9N0g4P22X2+0U03X6YH9rzWEMKAM9MHK+3sPcl/gEtvr2h5qex3gMVKSGAEcTqpVAkwBdpC0Uu6Msk/dM9Zh+2XgMUn7ASjZNG97JZ/7LOBq23NqDn8QWE3SB/Kxi0vaKG+7FTgYeNh2Nymhf4TU4agnpwPfl7RKPt9wYG/g/Lz9cVLCpOYzrgA8lV+PbnD+WcByDbZXXAccXXmT4wghhNKYQ3fTSxn1N0GOAq6qWXclqanzamD3/BPbT5GaMe8Cbiclkpf6UNZBwBhJ04H7SPc+Ky4jJbrLag+y/QYpkX8/HzuN1FyL7cdJNczKuHy3kWqqL/YUhO2JwM+B2yU9ko/Zy/a/8i4nA2dJ6gKqk/XpwHclTaVxzX0GMEfSdElfarDfF4DNJc2QdD9wRIN9QwihcLabXspIRQZWuUeWa5BXARfYrk2wA0b+HBeSLjQOdln/ynOVPb4QQnks0H2xajut/cGmv3NuevL6fpfXakWPpHNS7sE5hNRE+NuCy28p27OBQzodRwghlFELZunoqEITpO35BomVdCKp52u1Cba/U0xU9ZU1rhBCGCgG+oTJhTaxho6LP3YIoVn9bvLcZq2dm/7Ouf2pGxf5JtYQQgiLiBgoIIQQQqhjoLdQRoIMIYTQFlGDDCGEEOqIXqwhhBBCHdHEGkIIIdTRgomQO6pVEyaHEEII8+jGTS/9IWllSddLejj/XKmH/d4h6TpJf5F0f544okeRIEMIIbRFURMmAycAN9geRpp4/oQe9rsYOMP2e4EtgWcbnTQSZAghhLbotpte+mlP4KL8+iJgr9odJG0ILGb7ekgzQdl+rdFJI0GGEEJoiwJrkKvbfia//gewep193g38W9JvJE2VdIakwY1OGp10QgghtEVfOulIGguMrVo1Lk/4Xtn+R+DtdQ49sfqNbUuql3EXA7YDNgP+TpoecTRp+sK6IkGGEEJoi740neZkOK7B9l172ibpn5LWsP2MpDWof2/xSWCa7UfzMb8F3k+DBBlNrCGEENqiwCbWicCh+fWhwO/q7DMFWFHSavn9zsD9jU4aCTKEEEJbFNhJ53vAByU9DOya3yNpc0k/A7A9BzgOuEHSTNJsJT9tdNKY7mrREn/sEEKz+j391DtX3azp75xHn5sa012FEEJYNHhRGElH0l6SLGmDdgUi6ZVm95G0pqQr2hVLLzFsK+kuSQ9IelDSkS0+/9dq3v8p/xwq6d5WlhVCCO00x91NL2XU7D3IUcBt+WfH2X7a9r79PY+kPtWgJb0d+BVwhO0NgG2AMZL2bkEskjQImCdB2t66v+cOIYROKGqouXbpNUFKWhbYFhgDHChpN0kTqrbvKOnq/HqMpIdyDeunkn7c4LzrSfqzpJmSvl2z7XhJUyTNkHRynWPfqk1JukPSRlXbbs43ZpeRdEGOZaqkPfP20ZImSrqRdLP2Ykl7VR3/y8q+dRwFjLd9D4Dt54AvA8fnY8dLeitxV9V4l5V0g6R78uetxDI010IvBu4ldTdeStI0Sb+sPkfN5x+cH3Kt/I4+29PvOYQQOsV200sZNVOD3BO41vZDwPPAi8BWkpbJ2w8ALpW0JvAN0nMl2wC9NceeBZxr+31AZQQEJH0IGEYaJ284MFLS9g3Ocxmwfz52DWAN212kh0dvtL0lsBNwRlXMI4B9be9ASkqj8/ErAFsDf+ihrI2Au2vWdQEb9vJZXwf2tj0ix/I/kio3pIcB59jeyPZhwH9sD7d9UIPzjQFesr0FsAVwuKT16u0oaaykLkld48b1+IhRCCG0XIG9WNuimQQ5Crg0v74U2A+4Fvh4bqL8KOmZky2BybZfsP0mMKHeyapsA/w6v76kav2H8jIVuIeUaIc1OM/lQKXWtj9QuTf5IeAESdOAm4EhwDvytuttvwBgezIwLD8bMwq40vbsXmLvKwGnSZoB/BFYi7lDIf3N9h19PN+HgE/lz3YnsAo9/I5sj7O9ue3Nx44dW2+XEEJoiwKfg2yLhvfgJK1MepjyfXnonsGkRwUOIzU3vgB02Z41t0LUJ/V+KwK+a/v8pk5gPyXpeUmbkGqzR1SdZx/bD9Z8pq2AV2tOczFwMHAg6bP15H5gJPM+hDqSVIsEmE2+6Mj3E5fI6w8CVgNG2n5T0uOkhE2dWJoh4Gjbkxbg2BBCKERZm06b1VsNcl/gEtvr2h5qex3gMVIiGAEcztza5RRgB0kr5ZrlPr2c+3ZSQoKUQComAZ/O9z6RtJakt/VyrstI9wJXsD2j6jxHV5oyJW3W4PjxwDEAthuNrPATYLSk4fmcqwDfAU7N2x8nJUyAPYDF8+sVgGdzctwJWLdBGW9KWrzBdkif7XOV/SS9u6r5OIQQSmFh78U6CriqZt2VpMR2NbB7/ontp4DTgLtIye9x4KUG5/4icFQe0WCtykrb15F6iv45b7sCWK6XOK/IMV1ete5UUoKaIek+5iax+dj+J/AX4MJGheTR4g8Gxkl6EHga+FFupoU0KsMOkqYDH2Bu7fCXwOb583wKeKBBMeNyzL9ssM/PSLXZe3JnpfOJZ1pDCCUz0O9BtnQkHUnL2n4l1yCvAi6wXZtgS0fS0sBMYITtRkm99rgjgc8B29t+sV3xtVA5/xWGEMqo3yPbrLTsu5r+znnxlUdKN5JOq8diPSl3HLmX1BT72xafv+Uk7UqqPZ7dl+QIYPsc2+8bIMkxhBAKNdCfg2z7WKySTiT1fK02wfZ32lpwP0j6MPD9mtWP2e73gAAdVs5/hSGEMup3jW75Zd7Z9HfOy68+WroaZAxWvmiJP3YIoVn9TljLLD206e+cV197vHQJMjp2hBBCaIuydr5pViTIEEIIbTHQWygjQYYQQmiLso6Q06xIkCGEENoiapAhhBBCHQM9Qbb6OchQburvkqfW6vd5BnoMZYmjDDGUJY4yxFCWOFoUQ7+9+cZTanZpRXmtFgky9FUZpgQpQwxQjjjKEAOUI44yxADliKMMMQx4kSBDCCGEOiJBhhBCCHVEggx9Na7TAVCOGKAccZQhBihHHGWIAcoRRxliGPBiqLkQQgihjqhBhhBCCHVEggwhhBDqiAQZmiZpJUmbdKjsbZpZFxZNedLzEFoqEmRoSNLNkpaXtDJwD/BTSf/bgVDObnJd20gaUmR5PZH0cUkd/39X0rp5wnEkLSVpuQ7EsLWk+4EH8vtNJZ1TdBy57EjSC5mO/08WSm8F2y8DnwAutr0VsGtRhUv6gKT/B6wm6diq5SRgcFFxZPdKul3S9yR9VNIKBZdfcQDwsKTTJW3QiQAkHQ5cAZyfV60N/LYDoZwJfBh4HsD2dGD7IgMoWZLu+EXLwiQSZOjNYpLWAPYHru5A+UsAy5LGDV6uankZ2LfIQGy/CxgFzAQ+CkyXNK3IGHIcBwObAX8Fxkv6s6SxBX8ZHgVsQ/o7YPth4G0Flv8W20/UrJpTcAgdT9JQqouWhUYMVh56cwowCbjd9hRJ7wQeLqpw25OByZLG2/5bUeXWI2ltUlLYDtgUuA+4rROx2H5Z0hXAUsAxwN7A8ZJ+ZLuIpuf/2n5DSkNoSloMOjK30ROStgYsaXHgi8Bfig7C9hOV30VWdJKGdNGyJXBnjulhSR25aFlYRIIMDdmeAEyoev8osE8HQllS0jhgKFX/bm3vXGAMfwemAKfZPqLAcuchaU9gNPAu4GJgS9vP5ntg91PMvdnJkr4GLCXpg8CRwO8LKLfWEcBZwFrAU8B1pERRpFIkacpz0bLQiIECQkOS3g2cC6xue+Pci3UP298uOI7pwHnA3VRdndu+u8AYNgW2JTWfvYNUk55s++dFxZDjGA9cYPuWOtt2sX1DATEMAsYAHyLN/DAJ+JkXwS8USauSkvSupN/FdcAXbT9fcBynA/8GPgUcTbpoud/2iUXGsTCJBBkakjQZOB443/Zmed29tjcuOI67bY8ssswe4liWlCS3Aw4GsL1ugeUPBv5oe6eiyiwjSWfToHZk+wsFhlMKcdHSetHEGnqztO27au6vzO5AHL+XdCRwFfDfykrbLxQVgKQuYEngT8CtwPZF3xe1PUdSt6QVbL9UZNkAkmbSODEV9ZxsV0Hl9KhsSdp2N/DTvIQWiAQZevOcpPXJXwSS9gWe6UAch+afx1etM/DOAmPY3fa/CiyvJ68AMyVdD7xaWVnQF/LHCiijV7Yv6nQMlCBJQ6kuWhY60cQaGsq9VscBWwMvAo8BB9t+vJNxdYKk1YHTgDVt7y5pQ+ADHbgHeWi99SVJGoWQ9EPbx0j6PXWSg+09OhBWR0hq2MTf6d7fA1kkyNAUScsAg2zP6lD5n6q33vbFBcbwf8CFwIm2N829BKfafl9RMVTFshTwDtsPFlzubba3lTSLeROTANtevqA4Rtq+W9IO9bbnx4PaHUMk6YVcNLGGuiQd28N6AGwXPdzcFlWvhwC7kIa+KyxBAqvavlzSVwFsz5ZU+PNukj4O/IA0iMJ6koYDpxTxhWx72/yzoyO0VPVeHm77rOptkr4ItD1BApfknz8ooKweleWiZWEUCTL0pFRDVNk+uvq9pBWBSwsO41VJqzD3fuz7gcI7ygAnkR4IvxnA9rTcFF4YSZfYPqS3dQU4lPSIRbXRdda1XEmSdGkuWhZGkSBDXbZP7nQMvXgVWK/gMo8FJgLrS7odWI2Ch7vL3rT9Uk3P4u6CY9io+k1ubi7sMRxJo4BPAu+UNLFq03JAYT2bs44l6WolumhZaESCDA0pzWAxhvSF+NZsFrY/XXAc1fd5BgPvBS4vMgbb9+R7Xu8hNV89aPvNImPI7pP0SWCwpGHAF0iPnrRdbl6ujKDzcmU18AapM1dR/kTqTb0q8D9V62cBM4oIoGRJGjp80bIwik46oSFJE0izFHySNC7rQcBfbH+x4DiqO2PMBv5m+8mCyt7Z9o2SPlFvu+3fFBFHVTxLAyeSHgiH9ED4t22/XlD5g0gPoBd6kVQnjo4OmpB7j64HfBc4oWrTLGCG7UKeF66+aAFeq6wmX7TY/moRcSyMIkGGhiRNtb2ZpBm2N8ljTd5q+/0diGV15nbWucv2swWVe7Ltb0m6sM5md6A2PcL2PUWWWSeGmZ3ovVsnjhuAT3Ri0IRcfilGNirLRcvCJppYQ28qTYj/lrQx8A86MK2RpP2BM0gdUwScLel421e0u2zb38ovP2O7E7M01PofSW8nTW10me17OxDDPZK2sD2lA2VX6+SgCR0f2agqjm5JW/S+Z+iLSJChN+MkrQR8ndRBZVngmx2I40Rgi0qtUdJqwB9JSaIoj0m6FrgMuLFTY1za3iknyP2B8yUtT0qURQ4gvxVwkKS/kRJT5ZGCokdt+U1eOqmjSbpKWS5aFhrRxBoGhNomvdykNL3IZr587+9jwIHACNIE0pfa7sickDmm9wFfBg6wvUSB5dYdvWVRHLWlLCMbSXqANAVapy9aFhqRIEND+XmuC0kdD35KSgwn2L6u4DjOADYBfp1XHQDMtP3lIuOoimclUjf+g2wPLrjs95I+/z6kWewvA64s6p5sTSxvY97ezX8vuPxhpE4yG9bEUehzoWUQFy2tN6jTAYTS+7Ttl0k9JlcBDgG+V3QQto8nPUawSV7GdSI5StpB0jmkeSmHkJo5i3YBad6/D9ve0fa5RSdHSXtIepg0Nu9k4HHg/4qMIbuQNF/pbGAn0shKvygyAEnDJF0h6X5Jj1aWImOAlAhzMvwP6ZGoyhIWUNyDDL2pPI3+EeBi2/ep5gn1oti+Mt/nWQxA0soFT3f1ODCV9Pzl8bZfbXxEe9j+QCfKrXEq8H5SD87NJO1Enh+zYEvZvkGScnI4SdLdFHuf/ELgW8CZpCR9GB2ofEjag/RM6JrAs8C6wF+oeT4yNC8SZOjN3ZKuIz3v9VVJy1H8qC1I+ixwMvB6Ll8UP93VJrk23RGSLre9v+af3qgT95retP28pEGSBtm+SdIPCyy/4r/5fvTDkj4PPEXqSFakMiRpKM9Fy0IjEmTozRhgOPCo7dfyWKSHVTZK2sj2fQXEcRywse3nCiirJ2+XdBWwuu2NJW0C7FFg79HK4AxlmJPx35KWBW4BfinpWap6cBboi8DSpNGETgV2Zu7coUUpQ5KG8ly0LDSik07oF0n32B5RQDnXkh4If63XndsXw2TShM3n294sr7vX9sadiqlTlKY/e51Uez0IWAH4pe3nOxpYB+TnD/8CrEhK0isAp9u+o+A4/gjsReq0tCqpmXUL21sXGcfCJBJk6JfKSDsFlLMZ6V7PncB/K+uLfNZM0hTbW1R/ZknTbA8vqPzq6Ywq94HNIjytkaSbqD8X484dCKej4qKl9aKJNfRXUVdY5wM3AjPpwD3Q7DlJ6zN3uqt9SQNmF6JM0xnVJOslgMWBVzuQpI+rej2E9OhLIWOgVpQlSdd0Giv0GcyFVSTIMFAsbrvuJM4FOor0qMkGkp4iPeJwUCcCkbQtMMz2hZJWBZaz/VhR5Vcn69yreU9SB5FCVc3JWHG7pLsKDqPjSRpKddGy0IgEGfrrjYLK+T9JY4HfM28TayGPeeRBqY+0vWtuyhpke1YRZdeJ5VvA5qRpty4kfRn+AtimE/HkIfd+m+M6obf9W0nSylVvB5Gmd1qhyBhKkqRLc9GyMIl7kKEhSTfY3qW3dQXEUa925CJHTJF0RydmMakTxzRgM+CeqnuhM4p8zEPzTv01iJSwdyj6Gc2afxezSbX6U4oc/q+HJP0j2+8pKoaeFNVHYGEVNchQl9JEyUsDq+Zh1SqdQpYH1io6HtvrFV1mHVOVJsadwLyDUhc9WPYbti2pci90mYLLB/h41evZpJF09iw6iJL8u6iuQVaS9Jiig+jhoqWQOUIXVpEgQ08+CxxDGpWjeu7Bl4EfdyKgPN1W7ZibFxcYwhDS2KfVnS9M8bNJXC7pfGBFSYcDnyaNk1sY24f1vld7SVqDdF94w7yqi/QITqG9NkuSpKEkFy0Lk2hiDQ1JOtr22SWI41vAjqQvw2uA3YHbbO/bybiqSfqq7e8WVNYHSePjCphk+/oiys1l70maQeS9eVUXuVlTBc2LKGkH0n3XC3P5kJo298rLKbYPKSCOUiTp0B6RIENDkpYAjgC2z6tuJn0BvNnjQe2JYyawKTDV9qaSVgd+YfuDRcbRSFGDJnSSpM+Rmg+/zNzEtDnwbdLsJl+zvWkBcdwFfNb21Jr1w0mj+1xlu60j6pQlSedYOn7RsjCKJtbQm3NI3cXPye8PIc2e8JmC4/iP06zps5UmCH4WWKfgGHrT1kHca7rxz6eg7vxfALap6T18o6SPA08CXyogBoBla5MjgO1pkv5J1XCIbXQGaajB6jgm5uEIpwNXFRBDo4uW0yWdBXyNdHEZ+igSZKhL0mK2Z5OGqqr+n+tGSdM7EFKXpBVJ99ruJs3i/ucOxNFIW5tjKt34JZ1KGqDgEuaOmrJGO8uuiWO+R2vyGKB/s31eQWFI0kq2X6xZuTIw23YRg0mUIUlDeS5aFjoxH2ToSeU5rjl59BgAJL0TmFN0MLaPtP3v/AX8QeDQ6o4iksowpU9R04DtYfsc27Nsv2z7XIrrjPGypPlqI3ldkc14ZwLXKc3PuVxediTNSXlmQTEo9/CuXVlkkgZ6vmgBirxoWehEDTL0pPJlfxxwk+ZOADuU4q6M67L9eJ3VlwCdvv83oaByXpV0EHApqdY6iuJm0vh/pGbEC5n7eMPmpBk0CptayfY4SU+TBgffiPR7uB/4tu3fFxRGJUkfx9ye3iOB71NckoZ80WJ7npadDly0LHSik06oS9KTwP/mt0sBg/PrOaT7gf9b98AOKeKBaEk/qrP6JaDL9u/aWXZNHENJHWK2ISWG24FjerhwaEf5q5N6blZq7fcDP7H9jyLK74t29yyW9DHSvb/qJH1GgUm6MuzgL0mdhea7aCly0ISFTSTIUJekZ0idceo2G9o+udiIGiuiB6mkccAGzK0p7kN6KHwV0nyZx7Sz/GYV+bhJgxiutL1PJ2PIcXS8Z3ERf4+BdNEykESCDHWV4YulLwpKkHeQOkPMye8XA24FtgVm2t6w0fFFKcPfrixDnJUhjjL8PXIcpbhoGUiik07oSVEdTlqliEHTV2LemeKXAVbOCfO/9Q/piDL87cpy5V2GOMrw9wAobNzihUUkyNCTQgcj742kGxqtK2gQ8dOBaZIulDQemAqckcdC/WMB5TerDEmhLMqQnMry9yhLHANG9GINdRU1jVRvyjRouu2fS7oG2DKv+prtp/Pr44uMpRdlSAqFxCBpG9u3N1hXVM/iRsrw9wgLIBJkKLvSDJou6ffAr4CJNbO3l02hSSFfuKxje0bV6q8UVPzZzP94z1vrbJ/W7gAkDbHdaNaMMiRpiETdZ9FJJwwIZRg0PY+9eQDwUWAK6TnEq3v5cmxHHB1/3ETSzcAepIvsu0lD/91u+9iCyv8AsDXp4qn6mcPlgb2LGA+2KpZHgH+SOmzdShpEv/DnD/PIOX/oaYACSR+yfV3BYQ1ocQ8yDBTnS/qCpCvy8nlJixcZgO3Jto8kdXY4H9iflBiKNgQYDjycl02AtYExkn5YUAwr2H4Z+ARwse2tgF0LKhtgCVKHqcWA5aqWl4FCZ3ix/S7SYA0zSRdP0/Ok1kU7AHhY0umSNqjdGMmx76KJNQwUpRg0XdJSpHn3DiA1440vsvxsE+Z93ORcqh43KSiGxfJUT/sDJxZU5ltsTwYmSxpv+29Fl19N0tqkQRu2Iw0Kfh9Q+MP5tg/OA/mPAsbnCbUvBH5te1bR8SwMIkGGUivToOmSLid10LmWdP9zDilRFq3yuEmlGe+tx00kFfW4ySnAJFKz6pQ8Ru/DBZVdbck8gMNQqr7PbO/c4xGt93dSk/tpto8osNz52H5Z0hWk0a+OAfYGjpf0o07fohiIIkGGsruLVFObI2l923+Fjg2a/nPgB8B+pCvzx4ArC44B5j5ucjOp48X2wGlFPm5iewJVnU9sP0oaWahoE4DzgJ/RgUH0s81ItfdPSjqBdKEw2fbPiwwizwk5GngXcDGwpe1nJS1NGlknEmQfRSedUGqVkVAk7Uxqzpxn0HTbNxUQw7tJzVajgOeAy4DjbK/b7rIbxLQGcx83mVL1uElR5b+b1MS9uu2NJW1CmmXk2wXHcbftkUWW2UMcy5KS5HbkQduL/veRn8294P+3d+cxdlZlHMe/P8aFRSugiOCKRqhaC6IYjUQERMQFkqqAohJFlEjAJWqCGgU0aoQYFwIpLig1KsUIgtG6UXBBLdAiBZdoAI1aoxgFxAXa/vzjvJe5nV7auZPOed/3+vskkzv3nTbn6UDmzFme57H9gxFfO9T2ZrnEsWWZIKPTulA0XdJGyhnfCbZ/2zy72XYrlUm6kG4i6SpK7ufSQSk3STfaXlQ5jtMpF6UuYaiaUc08XknXAg8Erqa5yVr7XFTSFPA92wfXHHfSZYs1um6Kct42M4drcHuxhiXAsZS2Xyso6R1t5pSdTTn7/IikttJNdrS9Strk27C+4vgDxzevw4UaTN2yakfY/mvF8TbTnD9vlPSQNlJMJlUmyOi6dbbPbDMA25cClzZnfEdRLj88vLk9eknt6/NDNzingEOAE4HPUXIAa7mtaaRtAEkvB9ZVHB8A23vVHnOE7SR9FtjT9hGSngw8u/YZJPBPYK2k7zLUH9T2qZXjmBjZYo1O60I3hlGa6jGvAI6xXb1u7Yh0k8tr/iBsLkmdT0nW/zvlwtKra/WkHIrjtaOe276wYgzfolzaeo/tfZsuL2tsP7VWDE0cx496bvsLNeOYJJkgo9Mk7dqVurBdMSPd5CKadBPbJ7cQy07Adm3l2Ukavpm5PaXI/mrb1YoFSLrG9gHDv8xJut72frViGIplB+Axtn9de+xJlC3W6LRMjiO1lm4iaWQpucFZZI1LU8NsnzIjjp0pZ7I13SXpoUxvNz+L6RzVappSc2dTqgztJWk/4EzbR9aOZVJkgozoiftIN1Hlm4u1LkbN1V1A7XPJtwOXAU+Q9GNgNyqXu2ucTtlZuBLA9vXNVnjMUSbIiP74FSWN4CVD6SZvqxmA7TNqjrc1TcrL4JxoCngSsLxmDLZXN4Xs96HccT9qIAAACJBJREFUbv617XtqxtC4x/btM24WjyxcHrOTCTKiPzqTbtL06TwBeArl7A8A26+vHMrZQ5+vB35n+w81BpZ0iO0rJC2Z8aW9JWH7azXiGHKTpFcBU5KeCJxKyc2MOUo3j4iesH2p7WOBhcBKhtJNJL2gcjjLgEcAhwNXUbqJVL+o06S8/Iqy9bsLcHfF4Q9qXl864uMlFeMYOIXyC8t/KYUkbqf8PxJzlFusET3WVrrJUAnAG2wvblqP/dD2s2rF0MRxNHAW5dxNlFJv77T91YoxTA06q7RJ0v62V2/9T8ZsZYKMiLFJWmX7mZJ+ALwZ+DOwqnb5vaajy2G2/9K8341Scq1mw+TfM51yc4Vb+qEqaSVlVf9V4CLbN7YRxyTJFmtEzMX5zer1vZQbnL+gdBmpbbvB5Nj4G/V/ri2kdFE5GbhF0jmSDqwcA81t5oOBv1IajK+V9N7acUySrCAjorcknUVpIP3l5tExwFrb72opnl2ATwDH2Z7a2p+fxzieCryLsvX+gLbi6LusICNibJLeImmBis9IWt3CRSFsv5NS8m5x83F+G5OjpIMknQtcR7nVe3QLMTxJ0umS1lJ6P15NuTwVc5QVZESMTdLPm7qjhwMnUbZal9nev6V4FjCUtla53dWtwBpK/mWbLch+QjkHXV67P+ikSh5kRMzFIP/yRcCFtm/SjAz1KkFIbwLOAP5DSYoX9dtdLbZ9R8XxRrL97LZjmDSZICNiLq6T9B1KWbfTJD2Ydqq2vANYZPu2FsYeeISkS4DdbS+StBg40vYHawwuabnto5ut1eEtQQG2vbhGHJMoW6wRMTZJ2wH7ATfb/kdTrPuRtm9ovv4U2zdViGMFsMT2v+Z7rC3EcBWlYfPSoW4eN9peVGn8PWyvk/TYUV+3/bsacUyirCAjYmy2NwKrh97/jZJiMbCM0qdyvp0GXC3pZ5QKMoN4ajYJ3tH2qhk7zOtrDW57XfOaiXAbywQZEfOh1nnkUuAKYC3tFea+TdITmG539XJgXa3BJd3J9Nbq4PtuprdYF9SKZdJkgoyI+VDr7Ob+tkf2qKzoZEqqyUJJf6T05zyu1uC2u96CrLcyQUZEn31L0huBy9l0i7VKmoekKeDNtp8vaSdKZZ/qRduH4jkQeKLtCyQ9DHiw7VvaiqfvckknIrY5ST+tUbhc0qgf/q5ZE7bWv3UWcbwfeAawj+29Je0JXGz7OS2H1luZICNibJK+P7N7yKhn/w8knQc8ErgYuLdIQO1+kJKuB54GrB66TXtD0jzmLlusETFrTaPkHYGHNXVHB5dCFlAmiTZiWgQ8mU0bN19YMYTtKTd4Dxl6ZqB2w+S7bVvS4LLQTpXHnziZICNiHG+iNOHdk6E0D+AO4JzawTTbis+jTJDfBI4AfgRUmyBtv25LX5d0mu0PVwhluaSlwM6STgReD3y6wrgTK1usETE2SafY/lQH4lgL7AusaWrD7g580fZhLYd2L0mra9WolXQY8ALKyv7btr9bY9xJlRVkRMzFUkmnAs9t3l9JqSRzT+U4/m17o6T1TcHyvwCPrhzD1lSrUdtMiJkUt5FMkBExF+cC929eAV4DnAe8oXIc10rambKVeB3wT+AnlWPYmnndpptRKGDzwVMoYM6yxRoRsybpfrbXD9pdzfjaZs8qx/Y4YMGgHmzzrEpN2C2RtGZwq3Sex/kApYLPMsqq9ThgD9vvm++xJ1UaJkfEOFY1rxua8moASHo8sKGdkArbtw5Pjo1lrQSzqYsrjXOk7XNt32n7DtvnAUdVGnsiZYs1IsYxOE97B7BS0s3N+8cBW7zN2ZJ5P/+T9MkRj28HrrX9ddsfmu8YGndJOg74CmXL9ZUM5WXG+DJBRsQ4dpM0qH26FJhqPt9ASVJf2UpU963GGdL2wEKmV4ovo9Rj3VfSwbbfWiEGgFcBn2g+DPy4eRZzlAkyIsYxBTyIzVdm9wP+X4tmLwaeY3sD3FtZ54fAgZQuI1XYvpUtbKlWzMecGJkgI2Ic62yf2XYQY7i7whi7UH5puL15vxOwq+0Nkv5733+tulcAmSDHkEs6ETGOajl9syHp+1t6VqmI+EeB6yVdIOnzwBrgrKbU2/cqjD9bnfpv1wdJ84iIWZO0a61WUluJY1ATdiWl1NxwTdgVthdWjmcP4JnN22ts/6nm+LNRs6LPpMgWa0TMWhcmx0ZnasJKuhz4EnCZ7S7fGs0KckxZQUZEb3WhJqykg4BjgBcD11DSLL5h+z9txjWTpHdXTDmZCJkgI6K3JD0AOIn2a8IiaYrS8upE4IW1S7xtLR+zZiyTIpd0IqLPzgWe3rwOPj+vdhCSdqDkP54EHAB8vnYMlHzM/YDfNB+LgUcBJ0j6eAvx9F7OICOidwY1YYEDZtR/vULSzyvHspxyQWcF5fxzA2XLtbZO5GNOkqwgI6KPulQT9rPA0cCdwAXAGcAvK8cA0/mYA/fmYwJdysfsjawgI6KPWq8JK2lvSr3TVwK3ARdR7nUcXGP8EQb5mFdSvj/PBT7UwXzM3sglnYjoHUl/AD7WvN2BTWvC/tv2x0b+xW0bw0bKFuYJtn/bPLvZ9uPne+wtxNT5fMw+yQoyIvqoCzVhlwDHUlawKyjpHa3lGvYoH7M3soKMiN7pUlWYZgvzKMpW6yHAhcAltr9TOY5e5GP2SSbIiOgdSWtsP63tOGaStAulKPgxtg9tKYZW8zEnSSbIiOidrtSE7ZomH/OllJXk/sDltk9tN6r+SppHRPROJsfNNfmYv6SsHs8Bjmf68lLMQSbIiIjJ0JV8zImRW6wRET3WwXzMiZEzyIiIHutiPuakyBZrRES/LQHWUfIxPy3pUNL7cZvICjIiYgJ0JR9zkmSCjIiYMF3Ix5wEmSAjIiJGyBlkRETECJkgIyIiRsgEGRERMUImyIiIiBH+B60jjRr8dFNlAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Recency Distribution"
      ],
      "metadata": {
        "id": "U3nk1RkjYT7p"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# plotting distribusi hari sejak pembelian terakhir\n",
        "plt.figure(figsize=(10,5))\n",
        "sns.histplot(x='recent_days', data=recency, bins=20)\n",
        "plt.title('Distribution Recency', weight = 'bold')\n",
        "plt.xlabel('Hari')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "UqdkiAF_9LJD",
        "outputId": "d481dd50-aeaa-4b1f-9511-03c8f34a586a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Recency Box Plot"
      ],
      "metadata": {
        "id": "pnoPzzHRMpg1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(x=recency['recent_days'])\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "Xy0LtTH4Cm4S",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 280
        },
        "outputId": "a4dde482-0610-4fa0-f6df-af13bf7d386f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_temp.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zTU5V72YwRD_",
        "outputId": "adaf7456-dc9b-4832-f280-e504dcb5c4c6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 100739 entries, 0 to 100738\n",
            "Data columns (total 28 columns):\n",
            " #   Column                           Non-Null Count   Dtype         \n",
            "---  ------                           --------------   -----         \n",
            " 0   order_id                         100739 non-null  object        \n",
            " 1   customer_id_x                    100739 non-null  object        \n",
            " 2   order_status_x                   100739 non-null  object        \n",
            " 3   order_purchase_timestamp_x       100739 non-null  datetime64[ns]\n",
            " 4   order_approved_at_x              100739 non-null  datetime64[ns]\n",
            " 5   order_delivered_carrier_date_x   100739 non-null  datetime64[ns]\n",
            " 6   order_delivered_customer_date_x  100739 non-null  datetime64[ns]\n",
            " 7   order_estimated_delivery_date_x  100739 non-null  datetime64[ns]\n",
            " 8   customer_unique_id               100739 non-null  object        \n",
            " 9   customer_zip_code_prefix         100739 non-null  int64         \n",
            " 10  customer_city                    100739 non-null  object        \n",
            " 11  customer_state                   100739 non-null  object        \n",
            " 12  payment_sequential               100739 non-null  int64         \n",
            " 13  payment_type                     100739 non-null  object        \n",
            " 14  payment_installments             100739 non-null  int64         \n",
            " 15  payment_value_x                  100739 non-null  float64       \n",
            " 16  customer_id_y                    100739 non-null  object        \n",
            " 17  order_status_y                   100739 non-null  object        \n",
            " 18  order_purchase_timestamp_y       100739 non-null  datetime64[ns]\n",
            " 19  order_approved_at_y              100739 non-null  datetime64[ns]\n",
            " 20  order_delivered_carrier_date_y   100739 non-null  datetime64[ns]\n",
            " 21  order_delivered_customer_date_y  100739 non-null  datetime64[ns]\n",
            " 22  order_estimated_delivery_date_y  100739 non-null  datetime64[ns]\n",
            " 23  order_purchase_timestamp         100739 non-null  object        \n",
            " 24  recent_days                      100739 non-null  int64         \n",
            " 25  order_approved_at                100739 non-null  int64         \n",
            " 26  payment_value_y                  100739 non-null  float64       \n",
            " 27  is_churn                         100739 non-null  bool          \n",
            "dtypes: bool(1), datetime64[ns](10), float64(2), int64(5), object(10)\n",
            "memory usage: 21.6+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Frequency Distribution"
      ],
      "metadata": {
        "id": "OE7Uwl1BMtj1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# plotting distribusi frequency\n",
        "\n",
        "plt.figure(figsize=(10,5))\n",
        "sns.histplot(x='frequency', data=frequency, bins=20)\n",
        "plt.title('Distribusi Frequency', weight='bold')\n",
        "plt.xlabel('Frequency')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "Y7q6ZfsTBt0_",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "outputId": "4bff5338-10a2-4764-a883-59df2eff53af"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Monetary Distribution"
      ],
      "metadata": {
        "id": "cdtDZu0-M35M"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# plotting distribusi total purchase\n",
        "\n",
        "plt.figure(figsize=(10,5))\n",
        "sns.histplot(x='payment_value', data=monetary, bins=20)\n",
        "plt.title('Distribusi Total Pembelian', weight = 'bold')\n",
        "plt.xlabel('Total Pembelian')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "yWCZwDaqDzHb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "outputId": "578785fd-4949-4681-84b6-987ddc545066"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_temp.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E4unhKCet7kx",
        "outputId": "471a3419-fa73-4c6d-fd29-ad65909f25a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 100739 entries, 0 to 100738\n",
            "Data columns (total 28 columns):\n",
            " #   Column                           Non-Null Count   Dtype         \n",
            "---  ------                           --------------   -----         \n",
            " 0   order_id                         100739 non-null  object        \n",
            " 1   customer_id_x                    100739 non-null  object        \n",
            " 2   order_status_x                   100739 non-null  object        \n",
            " 3   order_purchase_timestamp_x       100739 non-null  datetime64[ns]\n",
            " 4   order_approved_at_x              100739 non-null  datetime64[ns]\n",
            " 5   order_delivered_carrier_date_x   100739 non-null  datetime64[ns]\n",
            " 6   order_delivered_customer_date_x  100739 non-null  datetime64[ns]\n",
            " 7   order_estimated_delivery_date_x  100739 non-null  datetime64[ns]\n",
            " 8   customer_unique_id               100739 non-null  object        \n",
            " 9   customer_zip_code_prefix         100739 non-null  int64         \n",
            " 10  customer_city                    100739 non-null  object        \n",
            " 11  customer_state                   100739 non-null  object        \n",
            " 12  payment_sequential               100739 non-null  int64         \n",
            " 13  payment_type                     100739 non-null  object        \n",
            " 14  payment_installments             100739 non-null  int64         \n",
            " 15  payment_value_x                  100739 non-null  float64       \n",
            " 16  customer_id_y                    100739 non-null  object        \n",
            " 17  order_status_y                   100739 non-null  object        \n",
            " 18  order_purchase_timestamp_y       100739 non-null  datetime64[ns]\n",
            " 19  order_approved_at_y              100739 non-null  datetime64[ns]\n",
            " 20  order_delivered_carrier_date_y   100739 non-null  datetime64[ns]\n",
            " 21  order_delivered_customer_date_y  100739 non-null  datetime64[ns]\n",
            " 22  order_estimated_delivery_date_y  100739 non-null  datetime64[ns]\n",
            " 23  order_purchase_timestamp         100739 non-null  object        \n",
            " 24  recent_days                      100739 non-null  int64         \n",
            " 25  order_approved_at                100739 non-null  int64         \n",
            " 26  payment_value_y                  100739 non-null  float64       \n",
            " 27  is_churn                         100739 non-null  bool          \n",
            "dtypes: bool(1), datetime64[ns](10), float64(2), int64(5), object(10)\n",
            "memory usage: 21.6+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Churn Rate Prediction"
      ],
      "metadata": {
        "id": "1tgZjKsUM_7J"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "churn = df_temp.groupby(['is_churn'])['customer_id_x'].count()\n",
        "plt.figure(figsize=(10,6))\n",
        "plt.title('Proporsi Persentase Jumlah Churn Customer', weight='bold')\n",
        "churn.plot.pie(autopct='%1.1f%%', startangle=90, explode = (0.1, 0), shadow=True)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "Jr8bCVBzFZig",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        },
        "outputId": "ef06541c-c2b0-4184-dfa8-81e6f1bba5e6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x432 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# plotting distribusi customer churn\n",
        "plt.figure(figsize=(15,8))\n",
        "sns.countplot(x='is_churn', data=df_temp)\n",
        "plt.title('Distribusi Customer Churn', weight='bold')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "SZvcHV22MKpx",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "outputId": "7424f23b-bb7a-4bfb-ff01-8ac31dee99c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Label Encoder"
      ],
      "metadata": {
        "id": "8rApEjnhBUqz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "encoder = LabelEncoder()\n",
        "dataset['customer_city']=encoder.fit_transform(dataset['customer_city'])\n",
        "dataset['customer_state']=encoder.fit_transform(dataset['customer_state'])"
      ],
      "metadata": {
        "id": "K7OL65m9BWhk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_full = dataset.merge(df_last_180[['customer_unique_id','purchased']],on='customer_unique_id')\n",
        "dataset_full.drop(columns='customer_unique_id',inplace=True)"
      ],
      "metadata": {
        "id": "bDhE0VkMBaTU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Machine Learning Models"
      ],
      "metadata": {
        "id": "2hVMfNBpBobE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(dataset_full.iloc[:,:-1],dataset_full.iloc[:,-1], test_size=0.2, random_state=31)"
      ],
      "metadata": {
        "id": "BRiiSFDUBrw_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# menghitung skor gini untuk models\n",
        "def Gini(y_true, y_pred):\n",
        "    # memerikasa dan mendapatkan jumlah sampel\n",
        "    assert y_true.shape == y_pred.shape\n",
        "    n_samples = y_true.shape[0]\n",
        "    \n",
        "    # mengurutkan baris pada kolom prediksi berdasarkan dari nilai tertinggi - terendah\n",
        "    arr = np.array([y_true, y_pred]).transpose()\n",
        "    true_order = arr[arr[:,0].argsort()][::-1,0]\n",
        "    pred_order = arr[arr[:,1].argsort()][::-1,0]\n",
        "    \n",
        "    # mencari Lorenz curves\n",
        "    L_true = np.cumsum(true_order) / np.sum(true_order)\n",
        "    L_pred = np.cumsum(pred_order) / np.sum(pred_order)\n",
        "    L_ones = np.linspace(1/n_samples, 1, n_samples)\n",
        "    \n",
        "    # Mencari Koefisien Gini (area antar kurva)\n",
        "    G_true = np.sum(L_ones - L_true)\n",
        "    G_pred = np.sum(L_ones - L_pred)\n",
        "    \n",
        "    # normalisasi koefisien Gini yang sebenarnya\n",
        "    return G_pred/G_true"
      ],
      "metadata": {
        "id": "6VXPkj0aCPzh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Mengevaluasi beberapa model Machine Learning dengan melatih data train dan data test\n",
        "def evaluate(X_train, X_test, y_train, y_test):\n",
        "    # Model yang digunakkan\n",
        "    model_name_list = ['Linear Regression',\n",
        "                      'Random Forest', 'Extra Trees',\n",
        "                       'Gradient Boosted','KNeighbors']\n",
        "\n",
        "    \n",
        "    # Perumpamaan model\n",
        "    model1 = LinearRegression()\n",
        "    model3 = RandomForestClassifier(n_estimators=50)\n",
        "    model4 = ExtraTreesClassifier(n_estimators=50)\n",
        "    model6 = GradientBoostingClassifier(n_estimators=20)\n",
        "    model7= KNeighborsClassifier(n_neighbors = 5)\n",
        "    \n",
        "    # hasil dari dataframe\n",
        "    results = pd.DataFrame(columns=['r2', 'accuracy','gini'], index = model_name_list)\n",
        "    \n",
        "    # Train dan predict dari setiap model\n",
        "    for i, model in enumerate([model1, model3, model4, model6,model7]):\n",
        "   \n",
        "        model.fit(X_train, y_train)\n",
        "        predictions = model.predict(X_test)\n",
        "        \n",
        "        # Metriks\n",
        "        r2 = r2_score(y_test,predictions)\n",
        "        preds=np.where(predictions>0.5,1,0)\n",
        "        accuracy = accuracy_score(y_test,preds)\n",
        "        gini=Gini(y_test,preds)\n",
        "        \n",
        "        # Memasukkan hasil kedalam dataframe\n",
        "        model_name = model_name_list[i]\n",
        "        results.loc[model_name, :] = [r2, accuracy,gini]\n",
        "    \n",
        "    return results"
      ],
      "metadata": {
        "id": "ChqBo7CBCRSq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Machine Learning Result Evaluation"
      ],
      "metadata": {
        "id": "DxOAhXZY1ZtX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "final = evaluate(X_train, X_test, y_train, y_test)\n",
        "final  "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "4M7afFtZCZj0",
        "outputId": "747e9d64-a6db-40f4-86ed-97808c7403ad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                         r2  accuracy      gini\n",
              "Linear Regression  0.067154  0.968502  0.041664\n",
              "Random Forest      0.840566  0.995136  0.867888\n",
              "Extra Trees        0.802606  0.993978  0.873124\n",
              "Gradient Boosted   0.169933  0.974678  0.256078\n",
              "KNeighbors         0.298998  0.978615  0.510283"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fc269c55-4b13-456f-87da-32d949bbe194\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>r2</th>\n",
              "      <th>accuracy</th>\n",
              "      <th>gini</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Linear Regression</th>\n",
              "      <td>0.067154</td>\n",
              "      <td>0.968502</td>\n",
              "      <td>0.041664</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Random Forest</th>\n",
              "      <td>0.840566</td>\n",
              "      <td>0.995136</td>\n",
              "      <td>0.867888</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Extra Trees</th>\n",
              "      <td>0.802606</td>\n",
              "      <td>0.993978</td>\n",
              "      <td>0.873124</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Gradient Boosted</th>\n",
              "      <td>0.169933</td>\n",
              "      <td>0.974678</td>\n",
              "      <td>0.256078</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>KNeighbors</th>\n",
              "      <td>0.298998</td>\n",
              "      <td>0.978615</td>\n",
              "      <td>0.510283</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fc269c55-4b13-456f-87da-32d949bbe194')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-fc269c55-4b13-456f-87da-32d949bbe194 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-fc269c55-4b13-456f-87da-32d949bbe194');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 104
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Conclusion"
      ],
      "metadata": {
        "id": "ASxWZtgIMMFd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Berdasarkan evaluasi model machine learning, didapatkan nilai r2 dari 5 model Machine Learning yaitu Linear Regression, Random Forest, Extra Trees, Gradient Boosted, KNeighbors. Hal tersebut menjelaskan bahwa yang mendekati nilai 1 berdasarkan analisis prediksi probabilitas pembelian pelanggan dari data Brazilian E-Commerce ini adalah model Random Forest Classifier dengan r2 = 0.84 , accuracy = 0.995 , dan gini impurity = 0.86 yang dapat disimpulkan merupakan model machine learning terbaik berdasarkan ke 5 model tersebut dalam pengujian dataset kali ini."
      ],
      "metadata": {
        "id": "x8zYmnWn1jP9"
      }
    }
  ]
}