predict = () => {
  var formdata = new FormData();
  formdata.append(
    "gender",
    $("input[name='btnradio_gender']:checked")[0].value.toString()
  );
  formdata.append(
    "married",
    $("input[name='btnradio_married']:checked")[0].value.toString()
  );
  formdata.append(
    "dependents",
    $("input[name='btnradio_dependents']:checked")[0].value.toString()
  );
  formdata.append(
    "education",
    $("input[name='btnradio_education']:checked")[0].value.toString()
  );
  formdata.append(
    "applicant_income",
    $("#applicant_income")[0].value.toString()
  );
  formdata.append(
    "coapplicant_income",
    ($("#coapplicant_income")[0].value | 0).toString()
  );
  formdata.append(
    "property_area",
    $("input[name='btnradio_area']:checked")[0].value.toString()
  );
  formdata.append(
    "self_employed",
    $("input[name='btnradio_self_employed']:checked")[0].value.toString()
  );
  formdata.append("loan_amount", $("#loan_amount")[0].value.toString());
  formdata.append(
    "loan_amount_term",
    $("#loan_term")[0].selectedOptions[0].value.toString()
  );
  formdata.append(
    "credit_history",
    $("input[name='btnradio_credit']:checked")[0].value.toString()
  );

  var requestOptions = {
    method: "POST",
    body: formdata,
  };

  fetch(`${window.location.href}/predict_loan`, requestOptions)
    .then((response) => response.text())
    .then((result) => {
      var status = JSON.parse(result)["approval_status"];
      if (status) {
        $(".notif-approved").removeAttr("hidden");
      } else {
        $(".notif-rejected").removeAttr("hidden");
      }
    })
    .catch((error) => console.log("error", error));
};

clear_notif = () => {
  $(".notif-alert").attr("hidden", "");
};

clear_notif();

$("#btn_submit").click(() => {
  clear_notif();
  predict();
});
